import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bottleneck import create_bottleneck 

###############################################################################
# 1. Base Convolution Blocks (1D)
###############################################################################
class DoubleConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


###############################################################################
# 2. Downsampling Block (1D)
###############################################################################
class DownBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=DoubleConvBlock1D, 
                 pool_type='max', pool_kernel=2):
        super().__init__()
        
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        elif pool_type == 'stride':
            self.pool = nn.Conv1d(in_channels, in_channels,
                                  kernel_size=3, stride=2, 
                                  padding=1, bias=False)
        else:
            raise ValueError("Invalid pool_type. Use 'max' or 'stride'")
        
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


###############################################################################
# 3. Upsampling Block (1D)
###############################################################################
class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=DoubleConvBlock1D, 
                 up_mode='transpose'):
        super().__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'interp':
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        else:
            raise ValueError("up_mode must be 'transpose' or 'interp'")
        
        self.conv_block = conv_block(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


###############################################################################
# 4. Flexible Exit Adapter (1D) -- no more dynamic shape logic
###############################################################################
class FlexibleExitAdapter1D(nn.Module):
    """
    Maps from in_channels -> out_channels, optionally with an up/down-sample
    by a fixed integer scale_factor.  We remove all dynamic shape logic.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample_type='interpolate',
                 scale_factor=1,
                 mode='linear'):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.upsample_type = upsample_type
        self.scale_factor = scale_factor
        self.mode = mode

        # If using transposed conv, create it statically (kernel_size=scale_factor, stride=scale_factor).
        if self.upsample_type == 'transpose':
            if not isinstance(self.scale_factor, int):
                raise ValueError("For 'transpose', scale_factor must be integer.")
            if self.scale_factor < 1:
                raise ValueError("scale_factor must be >= 1.")
            # if scale_factor == 1, it just acts as a 1x1 “learned identity”
            self.learned_upsampler = nn.ConvTranspose1d(
                out_channels,  # in_channels for the transposed conv = out_channels from conv1x1
                out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor
            )
        else:
            self.learned_upsampler = None

    def forward(self, x):
        # 1) adapt channels
        x = self.conv1x1(x)

        # 2) up/down sample if needed
        if self.upsample_type == 'none' or self.scale_factor == 1:
            # do nothing more
            return x
        elif self.upsample_type == 'interpolate':
            return F.interpolate(x, scale_factor=self.scale_factor, 
                                 mode=self.mode, align_corners=False)
        elif self.upsample_type == 'transpose':
            return self.learned_upsampler(x)
        else:
            raise ValueError(f"Unknown upsample_type={self.upsample_type}")


###############################################################################
# 5. UNet for 1D with multi-exits, no dynamic adapter creation
###############################################################################
class Model(nn.Module):
    """
    A 1D UNet with flexible multi-exits, where all adapters are fixed at init.
    """
    def __init__(self, config,
                 conv_block=DoubleConvBlock1D,
                 enc_block=DownBlock1D,
                 dec_block=UpBlock1D,
                 up_mode='transpose',
                 resize_mode='linear'):
        super().__init__()
        self.cfg = config['model']
        in_channels = self.cfg['in_channels']
        self.num_stages = num_stages = len(self.cfg['filters'])
        self.bottleneck = None
        if 'bottleneck' in self.cfg:
            # Only import/create if config has a bottleneck
            self.bottleneck = create_bottleneck(self.cfg)
        filters = self.cfg['filters']  
        exit_out_channels = self.cfg['exit_out_channels']
        
        self.exit_at_stages = self.cfg['exit_at_stages'] if self.cfg['exit_at_stages'] else []
        self._resize_mode = resize_mode

        # -----------------------------
        # Encoder (Down)
        # -----------------------------
        self.initial_block = conv_block(in_channels, filters[0])
        self.encoders = nn.ModuleList()
        for i in range(1, num_stages):
            self.encoders.append(
                enc_block(filters[i-1], filters[i], conv_block=conv_block)
            )
        
        # -----------------------------
        # Decoder (Up)
        # -----------------------------
        # reversed_filters: e.g. if filters=[64,128,256,512,512,512], reversed=>[512,512,512,256,128,64]
        reversed_filters = filters[::-1]
        self.decoders = nn.ModuleList()
        # we only need num_stages-1 up-blocks
        for i in range(num_stages - 1):
            self.decoders.append(
                dec_block(in_channels=reversed_filters[i],
                          out_channels=reversed_filters[i+1],
                          conv_block=conv_block,
                          up_mode=up_mode)
            )
        
        # Final 1x1 conv after the last decoder
        self.final_conv = nn.Conv1d(reversed_filters[-1], exit_out_channels, kernel_size=1)

        # -----------------------------
        # Build the exit adapters *now*, no dynamic creation
        # -----------------------------
        # We'll store them in a dictionary keyed by stage index
        #   - stage i in [0..(num_stages-2)] => the i-th decoder block
        #   - stage = (num_stages-1) => the "final" stage (post-final_conv).
        #
        # Upsample factor logic:
        #   - for stage i < (num_stages-1), scale_factor = 2^((num_stages-2) - i)
        #   - for the final stage i = (num_stages-1), scale_factor = 1
        #
        self.exit_adapters = nn.ModuleDict()
        for stage_idx in self.exit_at_stages:
            if stage_idx < (num_stages - 1):
                # this stage corresponds to decoders[stage_idx]
                in_ch = reversed_filters[stage_idx + 1]  # output of that decoder
                factor = 2 ** ((num_stages - 2) - stage_idx)
            else:
                # final stage: after final_conv
                in_ch = exit_out_channels
                factor = 1

            # If factor==1, we can skip upsampling
            if factor == 1:
                up_type = 'none'
            else:
                up_type = up_mode  # e.g. 'transpose' or 'interp'

            adapter = FlexibleExitAdapter1D(
                in_channels=in_ch,
                out_channels=exit_out_channels,
                upsample_type=up_type,
                scale_factor=factor,
                mode=self._resize_mode
            )
            self.exit_adapters[str(stage_idx)] = adapter

    def forward(self, x):
        # -----------------------------
        # Encoder
        # -----------------------------
        x0 = self.initial_block(x)
        skip_features = [x0]
        for enc in self.encoders:
            skip_features.append(enc(skip_features[-1]))
        bottleneck = skip_features[-1]

        # -----------------------------
        # Bottleneck
        # -----------------------------
        if self.bottleneck is not None:
            bottleneck = self.bottleneck(bottleneck)

        # -----------------------------
        # Decoder
        # -----------------------------
        out = bottleneck
        intermediate_exits = []
        num_decoders = len(self.decoders)  # = num_stages-1
        
        # i-th decoder reconstructs from skip_features[...] up one level
        for i, dec in enumerate(self.decoders):
            # skip index goes backwards
            skip_idx = (len(skip_features) - 2) - i
            skip = skip_features[skip_idx]
            out = dec(out, skip)

            # If we have an exit at this decoder stage, generate it
            if str(i) in self.exit_adapters:
                exit_out = self.exit_adapters[str(i)](out)
                intermediate_exits.append(exit_out)

        # -----------------------------
        # Final output
        # -----------------------------
        out = self.final_conv(out)

        # If we have an exit at the final stage (num_stages-1), generate it
        final_stage_idx = self.num_stages - 1
        if str(final_stage_idx) in self.exit_adapters:
            exit_out = self.exit_adapters[str(final_stage_idx)](out)
            intermediate_exits.append(exit_out)

        return {'output': out, 'exits': intermediate_exits}

    def compute_loss(self, model_output, targets, is_training=True):
        """
        Example loss computation with optional weighting of exits, ensuring sum of weights = 1.
        If 'exit_weights' is not found in config, use equal weighting.
        """
        final_pred = model_output['output']
        intermediate_exits = model_output['exits']
        all_preds = [final_pred] + intermediate_exits

        # 1) Fetch or define weights
        #    - If exit_weights is not in config, default to equi-weighted.
        exit_weights = self.cfg['loss'].get('exit_weights', None)
        if exit_weights is None:
            # No weights provided -> equi-weighted
            num_preds = len(all_preds)
            exit_weights = [1.0 / num_preds] * num_preds
        else:
            # If provided, make sure length matches number of predictions:
            if len(exit_weights) != len(all_preds):
                raise ValueError(
                    f"len(exit_weights)={len(exit_weights)}, "
                    f"but number of predictions={len(all_preds)}. They must match."
                )
            # Normalize so they sum to 1
            w_sum = sum(exit_weights)
            if w_sum == 0:
                # Edge case: if all weights are zero, revert to equi-weighted
                num_preds = len(all_preds)
                exit_weights = [1.0 / num_preds] * num_preds
            else:
                exit_weights = [w / w_sum for w in exit_weights]

        # 2) Other loss-related settings
        softness = self.cfg['loss']['softness']
        limit_db = self.cfg['loss']['limit_db']
        per_sample = self.cfg['loss']['per_sample']

        # 3) Validation loss logic
        if not is_training:
            # Weighted MSE across all preds, then convert to dB, clamp
            weighted_mse = 0.0
            for w, p in zip(exit_weights, all_preds):
                weighted_mse += w * (p - targets).pow(2).mean()
            mse_db = 10.0 * torch.log10(weighted_mse + 1e-12)
            return mse_db.clamp_min(limit_db)

        # 4) Training loss logic
        if per_sample:
            # Per-sample MSE -> dB -> soft clamp -> weighted sum -> average
            weighted_db_per_exit = []
            for w, p in zip(exit_weights, all_preds):
                mses = (p - targets).pow(2).mean(dim=list(range(1, p.ndim))) + 1e-12
                dbs = 10.0 * torch.log10(mses)
                # Weight the dB values
                weighted_db_per_exit.append(w * dbs)

            # Combine along exit dimension, then average over batch dimension
            db_all = torch.stack(weighted_db_per_exit, dim=0).sum(dim=0)
            return (limit_db + softness * torch.log1p(torch.exp((db_all - limit_db) / softness))).mean()
        else:
            # Weighted MSE -> dB -> soft clamp
            weighted_mse = 0.0
            for w, p in zip(exit_weights, all_preds):
                weighted_mse += w * (p - targets).pow(2).mean()
            avg_db = 10.0 * torch.log10(weighted_mse + 1e-12)
            return limit_db + softness * torch.log1p(torch.exp((avg_db - limit_db) / softness))

            """
            Example loss computation with weighted contribution from each exit,
            where exit_weights are always normalized to sum to 1.
            """
            final_pred = model_output['output']
            intermediate_exits = model_output['exits']
            all_preds = [final_pred] + intermediate_exits

            # Suppose your config includes something like:
            #   config['model']['loss']['exit_weights'] = [0.5, 0.3, 0.2]
            # and len(exit_weights) == len(all_preds)
            exit_weights = self.cfg['loss']['exit_weights']

            # Normalize weights so they sum to 1.0 (default to uniform if sum is too small).
            w_sum = sum(exit_weights)
            if w_sum < 1e-12:
                # Fall back to uniform weights if the sum is too close to zero
                normalized_weights = [1.0 / len(all_preds)] * len(all_preds)
            else:
                normalized_weights = [w / w_sum for w in exit_weights]

            # Fetch other loss settings
            softness = self.cfg['loss']['softness']
            limit_db = self.cfg['loss']['limit_db']
            per_sample = self.cfg['loss']['per_sample']

            # ---------------------------
            # Validation loss
            # ---------------------------
            if not is_training:
                # Weighted average MSE across all preds, then convert to dB, clamp
                mse_sum = 0.0
                for w, p in zip(normalized_weights, all_preds):
                    mse_sum += w * (p - targets).pow(2).mean()
                mse_db = 10.0 * torch.log10(mse_sum + 1e-12)
                return mse_db.clamp_min(limit_db)

            # ---------------------------
            # Training loss
            # ---------------------------
            if per_sample:
                # Per-sample MSE -> dB -> soft clamp -> weighted average
                db_values_for_each_exit = []
                for w, p in zip(normalized_weights, all_preds):
                    # MSE over all non-batch dimensions
                    mses = (p - targets).pow(2).mean(dim=list(range(1, p.ndim))) + 1e-12
                    dbs = 10.0 * torch.log10(mses)
                    db_values_for_each_exit.append(w * dbs)

                # Combine across exits
                db_all = torch.stack(db_values_for_each_exit, dim=0).sum(dim=0)
                return (limit_db + softness * torch.log1p(torch.exp((db_all - limit_db) / softness))).mean()
            else:
                # Weighted MSE -> dB -> soft clamp
                mse_sum = 0.0
                for w, p in zip(normalized_weights, all_preds):
                    mse_sum += w * (p - targets).pow(2).mean()

                avg_db = 10.0 * torch.log10(mse_sum + 1e-12)
                return limit_db + softness * torch.log1p(torch.exp((avg_db - limit_db) / softness))

