import torch.nn.functional as F

def my_loss_func(pred, target):
    # Example MSE loss
    return F.mse_loss(pred, target)
