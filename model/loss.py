""" 

Content: definition of the various loss-related modules and functions used throughout the torch_code/ directory.

Includes:
- `ME()` : (weighted) Mean Error;
- `RMSE()` : (weighted) Root Mean Squared Error;
- `MSE()` : (weighted) Mean Squared Error;
- `CE()` : (weighted) Entropy loss;
- `MAE()` : (weighted) Mean Absolute Error;
- `TrainLoss` : Module for the training loss.

"""

import torch
import torch.nn as nn
import math
import pickle
from os.path import join
import numpy as np

class QualityLoss(nn.Module):
    """
        Loss for mixing dense labels and sparse labels.
    """

    def __init__(self, mode, cutoff = 0.5, loss = 'MSE', version = 1):
        super(QualityLoss, self).__init__()

        assert mode in ['linear', 'log', 'exp', 'inv_log', 'random', 'false'], f"Invalid mode for QualityLoss: {mode}"
        assert cutoff in [0.10, 0.25, 0.5], f"Invalid cutoff for QualityLoss: {cutoff}"

        self.mode = mode
        self.cutoff = cutoff
        self.version = version

        # Set the reduction strategy for the loss functions
        if mode in ['random', 'false'] : reduction = 'mean'
        else: reduction = 'none'

        # Set the loss function
        if loss == 'MSE' : self.loss_fn = nn.MSELoss(reduction = reduction)
        elif loss == 'Huber' : self.loss_fn = nn.HuberLoss(reduction = reduction)
        else: raise ValueError(f"Invalid loss function: {loss}")
    
    def __call__(self, epoch, predictions, dense_labels, sparse_labels, center_x, center_y):
        """
        Args:
        - predictions: torch.Tensor, shape (batch_size, patch_size, patch_size)
        - dense_labels: torch.Tensor, shape (batch_size, patch_size, patch_size)
        - sparse_labels: torch.Tensor, shape (batch_size)
        - center_x: int, x-coordinate of the center pixel
        - center_y: int, y-coordinate of the center pixel
        """
        # predictions, shape (batch_size, patch_size, patch_size)
        # dense_labels, shape (batch_size, patch_size, patch_size)
        # sparse_labels, shape (batch_size)
        # mode: linear, log, or exp
        
        # Get the "central" pixel of the dense labels
        central_dense_labels = dense_labels[:, center_x, center_y]
        if torch.isnan(central_dense_labels).any():
            raise ValueError(f"NaN value in central_dense_labels")

        # Get the "central" pixel of the predictions
        central_predictions = predictions[:, center_x, center_y]
        if torch.isnan(central_predictions).any():
            raise ValueError(f"NaN value in central_predictions")

        # Calculate the per-sample difference between the dense and sparse labels
        # add a constant for numerical stability
        diff = torch.abs(sparse_labels - central_dense_labels) / (sparse_labels + 1e-6)
        diff[diff > self.cutoff] = self.cutoff

        # Calculate the scaling coefficient, alpha
        if self.mode == 'linear' : 
            alpha = - (1 / self.cutoff) * diff + 1
        elif self.mode == 'log' :
            cst = math.exp(1)
            alpha = torch.log((1 / self.cutoff) * diff * (1 - cst) + cst)
        elif self.mode == 'exp' :
            cst = math.log(2)
            alpha = 2 - torch.exp((1 / self.cutoff) * diff * cst)
        elif self.mode == 'inv_log' :
            cst = math.exp(-1)
            alpha = - torch.log((1 / self.cutoff) * diff * (1 - cst) + cst)
        elif self.mode == 'random' :
            alpha = 0.5
        elif self.mode == 'false' :
            alpha = 1

        # Case where alpha is a scalar
        if not torch.is_tensor(alpha) :
            dense_loss = alpha * self.loss_fn(predictions, dense_labels)
            sparse_loss = (1 - alpha) * self.loss_fn(central_predictions, sparse_labels)
            overall_loss = dense_loss + sparse_loss

        # Case where alpha is a tensor
        else:

            # Check if alpha has any NaN values
            if torch.isnan(alpha).any(): raise ValueError(f"NaN value in alpha")

            # v1
            if self.version == 1:
                dense_loss = alpha * torch.mean(torch.pow(predictions - dense_labels, 2))
                sparse_loss = (1 - alpha) * torch.mean(torch.pow(central_predictions - sparse_labels, 2))
            
            # v2
            elif self.version == 2:
                
                # Check if there is a NaN value in the predictions or in the dense labels
                if torch.isnan(predictions).any() or torch.isnan(dense_labels).any():
                    raise ValueError(f"NaN value in predictions or dense labels")
                
                # Calculate the losses
                dense_loss = alpha * self.loss_fn(predictions, dense_labels).mean(dim = (1,2)) # (batch_size, )
                sparse_loss = (1 - alpha) * self.loss_fn(central_predictions, sparse_labels) # (batch_size, )

            else: raise ValueError(f"Invalid version for QualityLoss: {self.version}")

            overall_loss = torch.mean(dense_loss + sparse_loss)

        return overall_loss


class BalancedLoss(nn.Module):
    """
    Loss for balancing the dense and sparse labels.
    """

    def __init__(self, _lambda, filter, path_h5, max_epochs = 15, loss = 'MSE', device = 'cuda'):
        super(BalancedLoss, self).__init__()

        # Set parameters
        self._lambda = _lambda
        self.filter = filter
        self.tmax = max_epochs - 1
        self.loss_fn = nn.MSELoss() if loss == 'MSE' else nn.HuberLoss()

        # Check valid parameters
        if not isinstance(_lambda, float) : assert _lambda in ['dense_gaussian', 'sparse_gaussian'], f"Invalid lambda for BalancedLoss: {_lambda}"
        assert filter in ['pct', 'R2_bin', 'none'], f"Invalid filter for BalancedLoss: {filter}"
        
        # Load the binned average values for R2_bin
        if filter == 'R2_bin' : 
            with open(join(path_h5, 'mean_agb_binned_values.pkl'), 'rb') as f:
                self.bin_avg = pickle.load(f)
                self.bin_avg = torch.tensor(np.array(list(self.bin_avg.values()))).to(device)

    def __call__(self, epoch, predictions, dense_labels, sparse_labels, center_x, center_y):
        """
        Args:
        - epoch: int, current epoch
        - predictions: torch.Tensor, shape (batch_size, patch_size, patch_size)
        - dense_labels: torch.Tensor, shape (batch_size, patch_size, patch_size)
        - sparse_labels: torch.Tensor, shape (batch_size)
        - center_x: int, x-coordinate of the center pixel
        - center_y: int, y-coordinate of the center pixel
        """

        # Get the batch size and patch size
        bs, ps, _ = predictions.shape

        # Take care of filter ---------------------------------------------------------------------

        # Get the "central" pixels of the dense labels
        central_dense_labels = dense_labels[:, center_x, center_y]

        # Get the "central" pixel of the predictions
        central_predictions = predictions[:, center_x, center_y]

        # Calculate the sparse loss
        sparse_loss = self.loss_fn(central_predictions, sparse_labels)

        # Calculate the error factor between the central dense labels and the sparse labels
        if self.filter == 'pct' :
            factor = 1 - torch.abs(sparse_labels - central_dense_labels) / (sparse_labels + 1e-6)
            factor = torch.clip(factor, 0, 1)
        elif self.filter == 'R2_bin' :
            bin_id = torch.div(sparse_labels, 50, rounding_mode = 'floor').type(torch.int)
            bin_avg = self.bin_avg[bin_id]
            factor = 1 - torch.pow(sparse_labels - central_dense_labels, 2) / (torch.pow(sparse_labels - bin_avg, 2) + 1e-6)
            factor = torch.clip(factor, 0, 1)
        elif self.filter == 'none' :
            factor = torch.ones(bs).to(predictions.device)

        # Calculate the dense loss
        SE = torch.sum(torch.pow(predictions - dense_labels, 2), dim = (1, 2)) # (bs, )
        dense_loss = torch.div(torch.sum(factor * SE), (bs * ps * ps))

        # Take care of lambda ---------------------------------------------------------------------
        if isinstance(self._lambda, float) : # constant lambda value
            overall_loss = sparse_loss + self._lambda * dense_loss
        else: # gaussian warmup
            _lambda = math.exp(-5 * (1 - math.pow(epoch / self.tmax, 2)))
            if self._lambda == 'dense_gaussian' :
                overall_loss = (1 - _lambda) * dense_loss + _lambda * sparse_loss
            elif self._lambda == 'sparse_gaussian' :
                overall_loss = (1 - _lambda) * sparse_loss + _lambda * dense_loss
        return overall_loss

class ME(nn.Module):
    """ 
        Weighted ME.
    """

    def __init__(self):
        super(ME, self).__init__()

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * (prediction - target))

class RMSE(nn.Module):
    """ 
        Weighted RMSE.
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.sqrt(torch.mean(weights * self.mse(prediction,target)))

class MSE(nn.Module):
    """ 
        Weighted MSE.
    """

    def __init__(self):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * self.mse(prediction,target))

class Huber(nn.Module):
    """ 
        Weighted Huber loss.
    """

    def __init__(self):
        super(Huber, self).__init__()
        self.huber = torch.nn.HuberLoss(reduction='none')

    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * self.huber(prediction, target))


class MAE(nn.Module):
    """ 
        Weighted MAE .
    """

    def __init__(self):
        super(MAE, self).__init__()
    
    def __call__(self, prediction, target, weights = 1):
        prediction = prediction[:, 0]
        return torch.mean(weights * torch.abs(prediction - target))

class GaussianNLL(nn.Module):
    """
        Gaussian negative log likelihood to fit the mean and variance to p(y|x)
        Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum over all samples N.
        Furthermore, the constant log term is discarded.
    """

    def __init__(self, eps = 1e-6):
        super(GaussianNLL, self).__init__()
        self.eps = eps

    def __call__(self, preds, target, weights = 1):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html 
        """
        prediction, log_variance = preds[:, 0], preds[:, 1]
        stable_variance = torch.exp(log_variance) + self.eps
        loss = 0.5 * (torch.log(stable_variance) + torch.pow(prediction - target, 2) / stable_variance)
        return torch.mean(weights * loss)


class TrainLoss(nn.Module):
    """ 
        Wrapper for the model's training loss.
    """

    def __init__(self, num_outputs, loss_fn):

        super(TrainLoss, self).__init__()
        self.task_num = num_outputs

        if loss_fn == 'MSE' :
            self.loss_fn = MSE()
        elif loss_fn == 'GNLL' :
            self.loss_fn = GaussianNLL()
        elif loss_fn == 'Huber' :
            self.loss_fn = Huber()
        else: raise ValueError(f"Invalid loss function: {loss_fn}")

    def forward(self, preds, labels, weights = 1):

        return self.loss_fn(preds, labels, weights)


class JSD(nn.Module):
    """
    This function calculates the Jensen-Shannon Divergence between two probability distributions.
    """

    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.softmax = nn.Softmax(dim = 1)
        self.eps = 1e-8

    def forward(self, preds, target):
        """
        Apply the Jensen-Shannon Divergence between two probability distributions.

        Args:
        - preds: torch.Tensor, shape (batch_size, patch_size, patch_size)
        - target: torch.Tensor, shape (batch_size, patch_size, patch_size)

        Returns:
        - jsd: float, Jensen-Shannon Divergence between the two distributions.
        """

        # Apply a Softmax over each patch to get a probability distribution
        ch_dist = self.softmax(target.flatten(start_dim=1)) + self.eps
        agb_dist = self.softmax(preds.flatten(start_dim=1)) + self.eps
        
        # Apply the JSDivergence
        m = (0.5 * (ch_dist + agb_dist)).log()
        return 0.5 * (self.kl(m, ch_dist.log()) + self.kl(m, agb_dist.log()))