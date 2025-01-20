# models/monai_unet.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet

class Model(nn.Module):
    def __init__(self, config):
        """
        Expects config.model to have at least:
          - spatial_dims
          - in_channels
          - out_channels
          - filters (list of channel sizes)
          - strides (list of strides, length = len(filters) - 1)
          - num_res_units
          - device (optional, fallback to 'cuda' or 'cpu')
        """
        super().__init__()
        cfg = self.cfg = config['model']

        # Use fallback device if not specified in config
        device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")

        self.unet = UNet(
            spatial_dims=cfg['spatial_dims'],      # e.g., 1 for 1D, 2 for 2D, etc.
            in_channels=cfg['in_channels'],        # e.g., 2
            out_channels=cfg['out_channels'],      # e.g., 2
            channels=cfg['filters'],               # e.g., [16, 32, 64, 128]
            strides=cfg['strides'],                # must match len(cfg.filters) - 1
            num_res_units=cfg['num_res_units'],    # e.g., 2
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Typical UNet forward pass
        unet_output = self.unet(x)
        # Return the raw UNet output, or you can do more logic here
        return {'output': unet_output}
    def compute_loss(self, model_output, targets, is_training=True):
        predictions = model_output['output']
        softness = self.cfg['loss']['softness']
        limit_db = self.cfg['loss']['limit_db']
        per_sample = self.cfg['loss']['per_sample']
        if not is_training:
            # Validation: average MSE, then hard clamp
            mse_db = 10 * torch.log10((predictions - targets).pow(2).mean() + 1e-12)
            return mse_db.clamp_min(limit_db)
        if per_sample:
            # Training: per-sample MSE -> dB -> soft clamp -> then average
            mses = (predictions - targets).pow(2).mean(dim=list(range(1, predictions.ndim))) + 1e-12
            dbs = 10 * torch.log10(mses)
            return (limit_db + softness*torch.log1p(torch.exp((dbs - limit_db)/softness))).mean()
        else:
            # Training: average MSE -> dB -> soft clamp
            avg_db = 10 * torch.log10((predictions - targets).pow(2).mean() + 1e-12)
            return limit_db + softness*torch.log1p(torch.exp((avg_db - limit_db)/softness))