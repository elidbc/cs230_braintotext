import torch
import torch.nn as nn
import time
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from dataset import BrainToTextDataset
from data_augmentations import gauss_smooth
import torchaudio.functional as F

class Exp1Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = config['experiment']['output_dir']

        # Params and Optimizer
        bias_params = [p for name, p in self.model.named_parameters() if 'bias' in name]
        adapter_params = [p for name, p in self.model.named_parameters() if 'day_adapter' in name]
        gru_params = [p for name, p in self.model.named_parameters() if 'gru_decoder' in name]
        classifier_params = [p for name, p in self.model.named_parameters() if 'classifier' in name]

        self.optimizer = torch.optim.AdamW([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': adapter_params, 'weight_decay': 0.0},
            {'params': gru_params, 'weight_decay': 0.0},
            {'params': classifier_params, 'weight_decay': 0.0},
        ])
        
        # Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = self.config['experiment']['num_training_batches'],
            eta_min = self.config['model']['lr_min'],
        )

        # Loss
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        self.transform_args = self.config['dataset']['data_transforms']        

    def _calculate_input_lengths(self, n_time_steps):
        """
        Calculate input lengths for CTC loss
        """
        patch_size = self.config['model']['patch_size']
        patch_stride = self.config['model']['patch_stride']
        adjusted_len = ((n_time_steps - patch_size) / patch_stride + 1).floor().to(torch.int32)
        return adjusted_len

    def transform_data(self, features, n_time_steps, mode = 'train'):
        """
        Apply augmentations and smoothing to data (inherited from baseline model)
        """
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
            
        return features, n_time_steps

    def train(self, train_loader, val_loader):
        """
        Train the model
        """
        print("Starting training...")
        
        for batch_idx, batch in enumerate(train_loader):
            # 1. Move data to device
            self.model.train()
            start_time = time.time()
            x = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            # 2. Apply data augmentations, 
            x, n_time_steps = self.transform_data(x, n_time_steps, 'train')
            adjusted_lens = self._calculate_input_lengths(n_time_steps)

            # 3. Forward pass -> phoneme predictions
            logits = self.model(x, day_indicies)

            # 4. Calculate CTC loss
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            loss = self.ctc_loss(
                log_probs = log_probs,
                targets = labels,
                input_lengths = adjusted_lens,
                target_lengths = phone_seq_lens,
            )

            # 5. Backward pass -> update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            self.scheduler.step()

            # 6. Log training progress
            if batch_idx % 100 == 0:
                print(f"Train batch {batch_idx}: loss {loss.item():.4f} time {time.time() - start_time:.2f}s")
            
            if batch_idx % 1000 == 0:
                val_per = self.validate(val_loader)
                self.save_checkpoint(batch_idx, val_per)

    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        print("Running validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)