import torch 
import torch.nn.functional as F
from torch import Tensor

# Reference : https://discuss.pytorch.org/t/elbo-loss-in-pytorch/137431
def ELBO_loss(recon_x: Tensor, x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    """calculate ELBO loss for VAE

    Args:
        recon_x (Tensor): prediction data, x_hat : (batch_size, seq_len, 2**num_classes)
        x (Tensor): input data, x : (batch_size, seq_len, 2**num_classes)
        mu (Tensor): (batch_size, encoder_latent_dim)
        std (Tensor): (batch_size, encoder_latent_dim)

    Returns:
        Tensor: loss 
    """
    log_var = std.pow(2).log()
    BCE = F.binary_cross_entropy(recon_x, x,  reduction='mean')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def get_accuracy(pred: Tensor, y: Tensor) -> Tensor:
    """calculate accuracy

    Args:
        pred (Tensor): prediction data, x_hat : (batch_size, seq_len, 2**num_classes)
        y (Tensor): input data x : (batch_size, seq_len, 2**num_classes)

    Returns:
        Tensor: accuracy
    """
    pred = torch.argmax(pred, dim=2)
    y = torch.argmax(y, dim=2)
    return torch.sum(pred == y) / (y.shape[0] * y.shape[1])