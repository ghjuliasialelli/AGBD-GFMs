""" 

Definition of the Model() Module, which is the wrapper module for all models.

"""

import numpy as np
import torch
import pytorch_lightning as pl
from loss import TrainLoss, RMSE, JSD
from torchmetrics.image import SpatialCorrelationCoefficient as SCC
from torchmetrics.clustering import NormalizedMutualInfoScore as MI

class Model(pl.LightningModule):

    def __init__(self, model, lr, step_size, gamma, patch_size, downsample, loss_fn,
                film = False, l2 = 0.0, crop = False, sim_dist = False,
                similarity = 'N/A', similarity_weight = 1.0, SCC_ws = 8, SCC_softmax = False,
                log_transform = False, sigreg_lambda = 0.0, predict = 'agbd'):
        """
        Initialize the Model() class.

        Args:
        - model (nn.Module): the model to train
        - lr (float): the learning rate
        - step_size (int): the step size for the learning rate scheduler
        - gamma (float): the gamma for the learning rate scheduler
        - patch_size (tuple): the size of the patches
        - downsample (bool): whether to downsample the images
        - loss_fn (str): the loss function to use
        - film (bool): whether to use FiLM
        - l2 (float): the L2 regularization parameter
        - crop (bool): whether to crop the images

        Returns:
        - None
        """
        
        super().__init__()
        self.model = model
        self.num_outputs = model.num_outputs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.best_val_rmse = np.inf
        self.film = film
        self.l2 = l2
        self.crop = crop
        self.loss_fn = loss_fn
        self.log_transform = log_transform
        self.sigreg_lambda = sigreg_lambda
        self.predict = predict

        # Similarity loss parameters
        self.sim_dist = sim_dist
        self.similarity = similarity
        self.similarity_weight = similarity_weight
        self.SCC_ws = SCC_ws
        self.SCC_softmax = SCC_softmax


        # Define the similarity loss
        if self.sim_dist :
            if similarity == 'N/A': raise ValueError('Similarity loss is enabled but similarity is not defined.')
            elif similarity == "SCC" : # bounded between 0 and 1, higher is better
                assert np.sign(similarity_weight) == -1, 'With the SCC, higher is better, so the similarity weight should be negative.'
                self.SimilarityLoss = SCC(window_size = self.SCC_ws)
            elif similarity == "JS" : # bounded between 0 and 1, lower is better
                assert np.sign(similarity_weight) == 1, 'With the JS, lower is better, so the similarity weight should be positive.'
                self.SimilarityLoss = JSD()
            elif similarity == "MI" :
                raise NotImplemented(f'Similarity loss {similarity} is not implemented.')
                self.SimilarityLoss = MI() # bounded between 0 and 1, higher is better
                # TODO this is supposed to be for clustering, not for regression, so it might not make sense
                # maybe do something like are they in the same bin, but then would need to align the distributions somehow
                assert np.sign(similarity_weight) == -1, 'With the MI, higher is better, so the similarity weight should be negative.'
            elif similarity == "KL" :
                raise NotImplemented(f'Similarity loss {similarity} is not implemented.')
                self.SimilarityLoss = KL() # unbounded, lower is better
                assert np.sign(similarity_weight) == 1, 'With the KL, lower is better, so the similarity weight should be positive.'
            else: raise NotImplemented(f'Similarity loss {similarity} is not implemented.')
        else: self.SimilarityLoss = None

        # With downsampling, we go from 10m per pixel to 50m per pixel
        if self.crop and downsample: raise Exception('Downsampling and cropping are not compatible.')
        if self.model.returns == 'pixel' : self.center_x, self.center_y = 0, 0
        else:
            if downsample: self.center_x, self.center_y = int(patch_size[0] // 5) // 2, int(patch_size[0] // 5) // 2
            else: self.center_x, self.center_y = int(patch_size[0] // 2), int(patch_size[0] // 2)
        
        # Placeholders for val/test preds/labels/biomes
        self.val_preds, self.test_preds = [], []
        self.val_labels, self.test_labels = [], []
        self.val_biomes, self.test_biomes = [], []

        if self.sigreg_lambda > 0.0 :
            self.val_latents = []
            self.test_latents = []

        # Loss function
        self.TrainLoss = TrainLoss(num_outputs = self.num_outputs, loss_fn = self.loss_fn)
        self.metric, self.ValLoss = 'rmse', RMSE()


    def compute_sigreg(self, x, global_step, num_slices=64):
        """
        Official SIGReg implementation for single GPU based on Algorithm 1.
        x: (N, K) tensor (Latents flattened to Batch*H*W, Channels)
        global_step: current training step for synchronized slice sampling
        """

        B, C, H, W = x.shape
        # x_flat shape: (B * H * W, C) -> (N, K) in Algorithm 1
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        dev = dict(device=x.device)
        
        # 1. Slice sampling (Fixed per step for stability)
        g = torch.Generator(**dev)
        g.manual_seed(global_step)
        proj_shape = (x.size(1), num_slices)
        A = torch.randn(proj_shape, generator=g, **dev)
        A /= A.norm(p=2, dim=0) # Normalize to unit sphere

        # 2. Epps-Pulley integration points
        # Creates 17 points between -5 and 5
        t = torch.linspace(-5, 5, 17, **dev)
        
        # 3. Theoretical Characteristic Function (CF) for N(0, 1)
        exp_f = torch.exp(-0.5 * t**2)
        
        # 4. Empirical CF
        # (N, M) @ (M, num_slices) -> (N, num_slices)
        # Then unsqueeze and multiply by t -> (N, num_slices, 17)
        x_t = (x @ A).unsqueeze(2) * t 
        
        # Calculate ECF using complex exponentials: exp(i * x_t)
        ecf = (1j * x_t).exp().mean(0) # Mean across the batch N
        
        # 5. Weighted L2 distance
        # Matching the empirical distribution to the Gaussian target
        err = (ecf - exp_f).abs().square().mul(exp_f)
        
        # 6. Numerical Integration (Trapezoidal rule)
        # N = x.size(0) here since world_size is 1
        N = x.size(0)
        T = torch.trapz(err, t, dim=1) * N
        
        return T.mean() # Return scalar loss
    
    def training_step(self, batch, batch_idx):

        # Parse the batch
        if self.crop: batch, (self.center_x, self.center_y) = batch[:-2], batch[-2:]
        if self.sim_dist: batch, ch = batch[:-1], batch[-1]

        # Get the model predictions
        if self.film :
            images, biomes, biome_embs, labels = batch
            predictions = self.model((images, biome_embs))
        else:
            images, biomes, labels = batch
            # predictions = self.model((images, biomes)) # TODO put back?
            predictions = self.model(images)
        if isinstance(predictions, tuple) : predictions, latents = predictions
                
        # Compare the CH distribution with the predictions' distribution
        if self.sim_dist: 
            if self.similarity in ['SCC', 'JS'] :
                preds, target = predictions[:, 0, :, :], ch
                if self.SCC_softmax :
                    N, W, H = preds.shape
                    preds = torch.softmax(preds.view(N, -1), dim=1).view(N, W, H) + 1e-8
                    target = torch.softmax(target.view(N, -1), dim=1).view(N, W, H) + 1e-8
            sim = self.SimilarityLoss(preds = preds, target = target)
            # check that it's not NaN or inf
            if torch.isnan(sim).any() or torch.isinf(sim).any(): raise Exception(f'Similarity loss is NaN or inf: {sim}')

        # Get the prediction at ground truth (GT) location
        if isinstance(self.center_x, int) :
            predictions = predictions[:, :, self.center_x, self.center_y]
        else: # the GT is different for each sample in the batch
            placeholder = torch.zeros((predictions.shape[0], predictions.shape[1])).to(self.device)
            for i, x, y in zip(range(predictions.shape[0]), self.center_x, self.center_y) :
                placeholder[i, :] = predictions[i, :, x, y]
            predictions = placeholder
        
        # Return the loss
        if self.sim_dist: # compare the CH distribution with the predictions' distribution
            loss = self.TrainLoss(predictions, labels) + self.similarity_weight * sim
        else: loss = self.TrainLoss(predictions, labels)

        if self.sigreg_lambda > 0.0 : loss = (1 - self.sigreg_lambda) * loss + self.sigreg_lambda * self.compute_sigreg(latents, self.global_step)

        # Log the train RMSE every 500 batches
        if batch_idx % 500 == 0:
            self.log(f'train/{self.predict}_{self.metric}', self.ValLoss.to(predictions.device)(predictions, labels).to(self.device), sync_dist = True)
            self.log('train/loss', loss.to(self.device), sync_dist = True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx = None):

        if self.crop: batch, (self.center_x, self.center_y) = batch[:-2], batch[-2:]
        if self.sim_dist: batch, _ = batch[:-1], batch[-1]

        # Ordinary validation
        if dataloader_idx == None or dataloader_idx == 0:
            
            # Get predictions
            if self.film :
                images, biomes, biome_embs, labels = batch
                predictions = self.model((images, biome_embs))
            else:
                images, biomes, labels = batch
                predictions = self.model(images)
            if isinstance(predictions, tuple) : predictions, latents = predictions

            # Log untransform
            if self.log_transform: 
                labels = torch.exp(labels) - 1
                predictions = torch.exp(predictions) - 1

            # Get the prediction at ground truth (GT) location
            if isinstance(self.center_x, int) :
                predictions = predictions[:, 0, self.center_x, self.center_y]
            else: # the GT is different for each sample in the batch
                placeholder = torch.zeros(predictions.shape[0]).to(self.device)
                for i, x, y in zip(range(predictions.shape[0]), self.center_x, self.center_y) :
                    placeholder[i] = predictions[i, 0, x, y]
                predictions = placeholder

            # Store the predictions, labels for the on_validation_epoch_end method
            self.val_preds.append(predictions.detach().cpu())
            self.val_labels.append(labels.detach().cpu())
            self.val_biomes.append(biomes.detach().cpu())
            if self.sigreg_lambda > 0.0 : self.val_latents.append(latents.detach().cpu())
        
        # Validation on the test set
        elif dataloader_idx == 1 :

            # Get predictions
            if self.film :
                images, biomes, biome_embs, labels = batch
                predictions = self.model((images, biome_embs))
            else:
                images, biomes, labels = batch
                predictions = self.model(images)
            if isinstance(predictions, tuple) : predictions, latents = predictions

            # Log untransform
            if self.log_transform: 
                labels = torch.exp(labels) - 1
                predictions = torch.exp(predictions) - 1

            # Get the prediction at ground truth (GT) location# Get the central pixel
            if isinstance(self.center_x, int) :
                predictions = predictions[:, 0, self.center_x, self.center_y]
            else: # the GT is different for each sample in the batch
                placeholder = torch.zeros(predictions.shape[0]).to(self.device)
                for i, x, y in zip(range(predictions.shape[0]), self.center_x, self.center_y) :
                    placeholder[i] = predictions[i, 0, x, y]
                predictions = placeholder

            # Store the predictions, labels for the on_validation_epoch_end method
            self.test_preds.append(predictions.detach().cpu())
            self.test_labels.append(labels.detach().cpu())
            self.test_biomes.append(biomes.detach().cpu())
            if self.sigreg_lambda > 0.0 : self.test_latents.append(latents.detach().cpu())
        
        else: raise ValueError('dataloader_idx should be 0 or 1')
    

    def on_validation_epoch_end(self):
        """
        Calculate the overall validation RMSE and binned metrics.
        """

        # Ordinary validation #####################################################################

        # Log the validation epoch's predictions and labels
        preds = torch.cat(self.val_preds).unsqueeze(1)
        labels = torch.cat(self.val_labels)
        val_agbd_rmse = self.ValLoss.to(preds.device)(preds, labels)
        biomes = torch.cat(self.val_biomes)
        self.log_dict({f'val/{self.predict}_{self.metric}': val_agbd_rmse.to(self.device), "step": self.current_epoch}, sync_dist = True)

        # Log the validation agbd rmse by bin
        bins = np.arange(0, 501, 50)
        for lb,ub in zip(bins[:-1], bins[1:]):
            pred, label = preds[(lb <= labels) & (labels < ub)], labels[(lb <= labels) & (labels < ub)]
            if len(pred) == 0 : continue
            rmse = self.ValLoss.to(preds.device)(pred, label)
            self.log_dict({f'binned/val_{self.metric}_{lb}-{ub}': rmse.to(self.device)}, sync_dist = True)
    
        # Log the validation agbd rmse by biome
        for biome in np.unique(biomes) :
            pred, label = preds[biomes == biome], labels[biomes == biome]
            if len(pred) == 0 : continue
            rmse = self.ValLoss.to(preds.device)(pred, label)
            self.log_dict({f'biomes/val_{self.metric}_{biome}': rmse.to(self.device)}, sync_dist = True)
        
        # Set the predictions and labels back to empty lists
            self.val_preds = []
            self.val_labels = []
        
        else:
            self.val_mses = []
            self.val_ns = []
        
        self.val_biomes = []

        # SIGReg loss
        if self.sigreg_lambda > 0.0 :
            latents = torch.cat(self.val_latents)
            val_sigreg_loss = self.compute_sigreg(latents, self.global_step)
            self.log_dict({'val/sigreg_loss': val_sigreg_loss.to(self.device), "step": self.current_epoch}, sync_dist = True)
            self.val_latents = []

        # Validation on the test set ##############################################################

        # Log the test set agbd rmse
        preds = torch.cat(self.test_preds).unsqueeze(1)
        labels = torch.cat(self.test_labels)
        test_agbd_rmse = self.ValLoss.to(preds.device)(preds, labels)
        biomes = torch.cat(self.test_biomes)
        self.log_dict({f'test/{self.predict}_{self.metric}': test_agbd_rmse.to(self.device)}, sync_dist = True)

        # Log the test set agbd rmse by bin
        bins = np.arange(0, 501, 50)
        for lb,ub in zip(bins[:-1], bins[1:]):
            pred, label = preds[(lb <= labels) & (labels < ub)], labels[(lb <= labels) & (labels < ub)]
            if len(pred) == 0 : continue
            rmse = self.ValLoss.to(preds.device)(pred, label)
            self.log_dict({f'binned/test_{self.metric}_{lb}-{ub}': rmse.to(self.device)}, sync_dist=True)
    
        # Log the test set agbd rmse by biome
        for biome in np.unique(biomes) :
            pred, label = preds[biomes == biome], labels[biomes == biome]
            if len(pred) == 0 : continue
            rmse = self.ValLoss.to(preds.device)(pred, label)
            self.log_dict({f'biomes/test_{self.metric}_{biome}': rmse.to(self.device)}, sync_dist = True)

        # Set the predictions and labels back to empty lists
            self.test_labels = []
            self.test_preds = []
        else:
            self.test_mses = []
            self.test_ns = []
        self.test_biomes = []

        # SIGReg loss
        if self.sigreg_lambda > 0.0 :
            latents = torch.cat(self.test_latents)
            sigreg_loss = self.compute_sigreg(latents, self.global_step)
            self.log_dict({'test/sigreg_loss': sigreg_loss.to(self.device), "step": self.current_epoch}, sync_dist = True)
            self.test_latents = []

        #########################################
        # Keep track of the best overall
        if self.sigreg_lambda > 0.0 : 
            val_loss = (1 - self.sigreg_lambda) * val_agbd_rmse + self.sigreg_lambda * val_sigreg_loss
            if val_loss < self.best_val_rmse:
                self.best_val_rmse = val_loss
                self.log_dict({'best_test_sigreg': val_loss.to(self.device)}, sync_dist = True)
        else:
            if val_agbd_rmse < self.best_val_rmse:
                self.best_val_rmse = val_agbd_rmse
                self.log_dict({f'best_test_{self.metric}': test_agbd_rmse.to(self.device)}, sync_dist = True)
    
    
    def backward(self, loss, *args, **kwargs):
        """
        To check that we don't have unwanted unused parameters.
        """
        # Call the default backward() method
        super().backward(loss, *args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2)
        return [optimizer], [torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.step_size, gamma = self.gamma)]
