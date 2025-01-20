import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1. ResNet-Style 1D Bottleneck
###############################################################################
class ResNetBottleneck1D(nn.Module):
    """
    A classic 1D ResNet-style bottleneck:
      1x1 conv (reduce) -> 3x3 conv -> 1x1 conv (expand)
      with a skip connection.
    """
    def __init__(self, channels, expansion=4):
        """
        Args:
            channels (int): The "output" channels of the bottleneck.
            expansion (int): Factor by which mid_channels is reduced or expanded.
                             Typically 4 for ResNet-like blocks.
        """
        super().__init__()
        # mid_channels is typically channels // expansion
        mid_channels = channels // expansion
        
        self.conv_reduce = nn.Conv1d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm1d(mid_channels)
        
        self.conv_mid = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, 
                                  padding=1, bias=False)
        self.bn_mid = nn.BatchNorm1d(mid_channels)
        
        self.conv_expand = nn.Conv1d(mid_channels, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # skip connection
        
        out = self.conv_reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)

        out = self.conv_mid(out)
        out = self.bn_mid(out)
        out = self.relu(out)

        out = self.conv_expand(out)
        out = self.bn_expand(out)

        # Add skip
        out += identity
        out = self.relu(out)
        return out


###############################################################################
# 2. MobileNetV2-Style Inverted Residual Bottleneck (1D)
###############################################################################
class InvertedResidualBottleneck1D(nn.Module):
    """
    A MobileNetV2-style 1D bottleneck:
      1x1 conv (expand) -> depthwise conv -> 1x1 conv (reduce)
      skip connection if stride=1 and in_channels==out_channels.
    """
    def __init__(self, channels, expansion_factor=4):
        """
        Args:
            channels (int): The "in/out" channels of the bottleneck
                            (assuming stride=1).
            expansion_factor (int): Factor for expansion step.
        """
        super().__init__()
        expanded_channels = channels * expansion_factor
        
        self.conv_expand = nn.Conv1d(channels, expanded_channels, 
                                     kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm1d(expanded_channels)
        
        # Depthwise convolution
        self.conv_depthwise = nn.Conv1d(expanded_channels, expanded_channels, 
                                        kernel_size=3, padding=1, 
                                        groups=expanded_channels, bias=False)
        self.bn_depthwise = nn.BatchNorm1d(expanded_channels)
        
        # Pointwise reduce
        self.conv_reduce = nn.Conv1d(expanded_channels, channels, 
                                     kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm1d(channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # Expand
        out = self.conv_expand(x)
        out = self.bn_expand(out)
        out = self.relu(out)
        
        # Depthwise
        out = self.conv_depthwise(out)
        out = self.bn_depthwise(out)
        out = self.relu(out)
        
        # Reduce
        out = self.conv_reduce(out)
        out = self.bn_reduce(out)
        
        # If input and output shapes match, we can add skip
        if out.shape == identity.shape:
            out = out + identity
        
        return out


###############################################################################
# 3. Factory function
###############################################################################
from omegaconf import ListConfig

def create_bottleneck(model_cfg):
    if 'bottleneck' not in model_cfg:
        return None

    bottleneck_cfg = model_cfg['bottleneck']
    bottleneck_type = bottleneck_cfg.get('type', 'resnet')

    # Convert ListConfig -> Python list for safety
    filters = model_cfg['filters']
    if isinstance(filters, ListConfig):
        filters = list(filters)

    if not isinstance(filters, (list, tuple)) or len(filters) == 0:
        raise ValueError("Model config must have a non-empty 'filters' list.")

    # Now 'filters' is a standard list, so the rest proceeds normally:
    channels = filters[-1]

    if bottleneck_type == 'resnet':
        expansion = bottleneck_cfg.get('expansion', 4)
        return ResNetBottleneck1D(channels=channels, expansion=expansion)
    elif bottleneck_type == 'mobilenetv2':
        expansion_factor = bottleneck_cfg.get('expansion_factor', 4)
        return InvertedResidualBottleneck1D(channels=channels,
                                            expansion_factor=expansion_factor)
    else:
        raise ValueError(f"Unsupported bottleneck type: {bottleneck_type}")

