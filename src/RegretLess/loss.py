import torch

def my_loss_func(pred, target, limit_db=-50.0):
    """
    Computes:
      1) Overall MSE (in dB) for the entire batch, then clamps it.
      2) Per-sample MSE (in dB), clamps each sample's MSE-dB, then averages.
    
    Returns a dictionary with two keys:
      {
        'mse_db_clamped': scalar,
        'mse_db_samplewise_clamped_mean': scalar
      }
    """

    # 1) "Overall" MSE -> single scalar across the entire batch
    overall_mse = (pred - target).pow(2).mean()
    overall_mse_db = 10.0 * torch.log10(overall_mse + 1e-12)
    overall_mse_db_clamped = overall_mse_db.clamp_min(limit_db)

    # 2) "Per-sample" MSE: clamp each sample's MSE-dB before averaging
    #    - We assume that the first dimension (dim=0) is the batch dimension.
    #    - For each sample, we average across all other dimensions of pred/target.
    #    - Then convert to dB, clamp, and compute the mean across the batch.
    #
    #    NOTE: You may need to adjust which dimensions you average over (e.g., dim=1,2,...) 
    #    depending on how your input tensor is shaped.
    #
    #    Example assumption:
    #      pred.shape = (batch_size, n_features, ...) 
    #      target.shape = (batch_size, n_features, ...)
    #    We'll compute sample-wise mean over all dimensions except batch (dim=0).
    dims_to_average = list(range(1, pred.ndim))
    per_sample_mse = (pred - target).pow(2).mean(dim=dims_to_average)  # shape: (batch_size,)
    
    per_sample_mse_db = 10.0 * torch.log10(per_sample_mse + 1e-12)
    per_sample_mse_db_clamped = per_sample_mse_db.clamp_min(limit_db)
    
    # Average the per-sample dB values after clamping
    mse_db_samplewise_clamped_mean = per_sample_mse_db_clamped.mean()

    # Return a dictionary with both metrics
    return {
        'mse_db_clamped': overall_mse_db_clamped,
        'mse_db_samplewise_clamped_mean': mse_db_samplewise_clamped_mean
    }
