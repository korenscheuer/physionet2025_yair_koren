import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wfdb
from tqdm import tqdm
import scipy
import pickle
import random

from gru_attentioned_resnet1d import ResNet1D # Ensure this import is correct and points to your modified resnet1d.py

import os
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, SubsetRandomSampler

########## Set fixed seed and specific generator for train_test split

def set_seed(seed=42):
    random.seed(seed)                     # Python RNG
    np.random.seed(seed)                  # NumPy RNG
    torch.manual_seed(seed)               # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)          # PyTorch GPU RNG (if using CUDA)
    torch.cuda.manual_seed_all(seed)      # All GPUs (if multi-GPU)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Call this at the start of the script

########## Constants
WINDOW_LENGTH = 1 # In seconds

# Configuration for the three datasets
datasets_config = {
    'code15': {'path': 'code15/', 'type': 'code15'},
    'ptbxl': {'path': 'ptbxl_100hz_patient_id/', 'type': 'ptbxl'},
    'sami': {'path': 'samitrop/', 'type': 'sami'}
}

# Original sampling frequencies for each dataset
original_fs_dict = {
    'code15': 400,
    'ptbxl': 100,
    'sami': 400
}

dataset_name_to_id = {'code15' : 0, 'ptbxl' : 1, 'sami' : 2}

########## Transformer
class ECGTransform(object):
    """
    This will transform the ECG signal into a PyTorch tensor. This is the place to apply other transformations as well, e.g., normalization, etc.
    """
    def __call__(self, signal, original_fs, new_fs, desired_signal_length):
        # Resampling the signal
        desired_signal_samples_number = desired_signal_length * original_fs
        clipped_signal = signal[:desired_signal_samples_number, :]
        num_samples = int(clipped_signal.shape[0] * new_fs / original_fs)
    
        # Resample each lead (column) independently
        resampled_signal = np.zeros((num_samples, clipped_signal.shape[1]))

        for i in range(clipped_signal.shape[1]):
            resampled_signal[:, i] = scipy.signal.resample(clipped_signal[:, i], num_samples)

        # Transform the data type from double (float64) to single (float32) to match the later network weights.
        signal_tensor = torch.tensor(resampled_signal)
        t_signal = signal_tensor.to(torch.float32)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        t_signal = torch.transpose(t_signal, 0, 1)
        return t_signal

########## Dataset
class UnifiedECGDataset(Dataset):
    def __init__(self, datasets_config, transform, original_fs_dict, new_fs, desired_signal_length, train_set=[]):
        """
        Unified dataset that combines multiple ECG datasets
        
        Args:
            datasets_config: dict with dataset names as keys and their config as values
                Example: {
                    'code15': {'path': 'code15/', 'type': 'code15'},
                    'ptbxl': {'path': 'ptbxl_100hz/', 'type': 'ptbxl'},
                    'sami': {'path': 'samitrop/', 'type': 'sami'}
                }
            transform: ECGTransform object
            original_fs_dict: dict mapping dataset names to their original sampling frequencies
                Example: {'code15': 400, 'ptbxl': 100, 'sami': 400}
            new_fs: target sampling frequency
            desired_signal_length: desired signal length in seconds
        """
        super().__init__()
        self.datasets_config = datasets_config
        self.transform = transform
        self.original_fs_dict = original_fs_dict
        self.new_fs = new_fs
        self.desired_signal_length = desired_signal_length
        self.train_set = train_set
        self.code15_train = []
        self.ptbxl_train = []
        self.sami_train = []
        self.divide_train_set()
        self.record_info = self.build_unified_index()

    def divide_train_set(self):
        for file in self.train_set:
            dataset = file.split('/')[0]
            if dataset == 'code15':
                self.code15_train.append(file)
            if dataset == 'ptbxl_100hz_patient_id':
                self.ptbxl_train.append(file)
            if dataset == 'samitrop':
                self.sami_train.append(file)

    def build_unified_index(self):
        """
        Builds a unified list of (record_path, start_sample_index, label, dataset_name)
        """
        unified_index = []
        
        for dataset_name, config in self.datasets_config.items():
            print(f"Processing {dataset_name}...")
            dataset_index = self._build_dataset_index(dataset_name, config)
            unified_index.extend(dataset_index)
            
        return unified_index

    def _build_dataset_index(self, dataset_name, config):
        """Build index for a specific dataset"""
        dataset_path = config['path']
        dataset_type = config['type']
        original_fs = self.original_fs_dict[dataset_name]
        
        if dataset_type == 'code15':
            return self._build_code15_index(dataset_path, dataset_name, original_fs)
        elif dataset_type == 'ptbxl':
            return self._build_ptbxl_index(dataset_path, dataset_name, original_fs)
        elif dataset_type == 'sami':
            return self._build_sami_index(dataset_path, dataset_name, original_fs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _build_code15_index(self, db_path, dataset_name, original_fs):
        """Build index for CODE-15 dataset"""
        with open('sig_len.pkl', 'rb') as f:
            signal_len_dict = pickle.load(f)

        index = []
        for file in self.code15_train:
            with open(file, 'rb') as f:
                record = pickle.load(f)
            label = int(record.comments[2].split()[2] == 'True')
            total_samples = record.p_signal.shape[0]

            window_samples = self.desired_signal_length * original_fs
            if label == 0:  # Negative: only 1 window starting at 0
                if total_samples // window_samples > 2:
                    max_start = total_samples - window_samples
                    start = random.randint(window_samples, max_start-window_samples)
                    index.append((file, start, label, dataset_name))
            else:  # Positive: add multiple windows
                n_windows = total_samples // window_samples
                for i in range(1, n_windows-1):
                    start = i * window_samples
                    index.append((file, start, label, dataset_name))

        return index

    def _build_ptbxl_index(self, db_path, dataset_name, original_fs):
        """Build index for PTB-XL dataset"""
        index = []
        window_samples = self.desired_signal_length * original_fs

        for file in self.ptbxl_train:
            # PTB-XL are all negatives, single window per record
            record = wfdb.rdrecord(file)

            total_samples = record.p_signal.shape[0]
            if total_samples // window_samples > 2:
                max_start = total_samples - window_samples
                start = random.randint(window_samples, max_start-window_samples)
                index.append((file, start, 0, dataset_name))

        return index

    def _build_sami_index(self, db_path, dataset_name, original_fs):
        """Build index for Sami-Trop dataset"""
        index = []
        window_samples = self.desired_signal_length * original_fs
        
        for file in self.sami_train:
            # Sami-Trop are all positives: add multiple windows like CODE-15 positives

            record = wfdb.rdrecord(file)
            total_samples = record.p_signal.shape[0]
            n_windows = total_samples // window_samples
            for i in range(1, n_windows-1):
                start = i * window_samples
                index.append((file, start, 1, dataset_name))

        return index

    def __getitem__(self, index):
        record_path, start_sample, label, dataset_name = self.record_info[index]
        original_fs = self.original_fs_dict[dataset_name]
        
        if dataset_name == 'code15':
            with open(record_path, 'rb') as f:
                record = pickle.load(f)
            signal_slice = record.p_signal[start_sample:start_sample + self.desired_signal_length * original_fs]
        else:  # ptbxl or sami
            record = wfdb.rdrecord(record_path)
            signal_slice = record.p_signal[start_sample:start_sample + self.desired_signal_length * original_fs]
        
        signal = self.transform(signal_slice, original_fs, self.new_fs, self.desired_signal_length)

        exam_id = -1 # -1 if sami sample
        if dataset_name == 'code15' or dataset_name == 'ptbxl':
            exam_id = int(record.record_name.split('_')[1])
        else:
            exam_id = int(record.record_name.split('.')[0])

        return signal, label, dataset_name_to_id[dataset_name], exam_id 

    def __len__(self):
        return len(self.record_info)

    def get_dataset_indices(self, dataset_name):
        """Get indices for a specific dataset"""
        return [i for i, (_, _, _, ds_name) in enumerate(self.record_info) if ds_name == dataset_name]

    def get_dataset_stats(self):
        """Get statistics for each dataset"""
        stats = {}
        for dataset_name in self.datasets_config.keys():
            indices = self.get_dataset_indices(dataset_name)
            labels = [self.record_info[i][2] for i in indices]
            stats[dataset_name] = {
                'total': len(labels),
                'positives': sum(labels),
                'negatives': len(labels) - sum(labels)
            }
        return stats

class UnifiedECGDatasetTestSet(Dataset):
    def __init__(self, datasets_config, transform, original_fs_dict, new_fs, desired_signal_length, test_set=[]):
        """
        Unified dataset that combines multiple ECG datasets
        
        Args:
            datasets_config: dict with dataset names as keys and their config as values
                Example: {
                    'code15': {'path': 'code15/', 'type': 'code15'},
                    'ptbxl': {'path': 'ptbxl_100hz/', 'type': 'ptbxl'},
                    'sami': {'path': 'samitrop/', 'type': 'sami'}
                }
            transform: ECGTransform object
            original_fs_dict: dict mapping dataset names to their original sampling frequencies
                Example: {'code15': 400, 'ptbxl': 100, 'sami': 400}
            new_fs: target sampling frequency
            desired_signal_length: desired signal length in seconds
        """
        super().__init__()
        self.datasets_config = datasets_config
        self.transform = transform
        self.original_fs_dict = original_fs_dict
        self.new_fs = new_fs
        self.desired_signal_length = desired_signal_length
        self.test_set = test_set
        self.code15_test = []
        self.ptbxl_test = []
        self.sami_test = []
        self.divide_train_set()
        self.record_info = self.build_unified_index()

    def divide_train_set(self):
        for file in self.test_set:
            dataset = file.split('/')[0]
            if dataset == 'code15':
                self.code15_test.append(file)
            if dataset == 'ptbxl_100hz_patient_id':
                self.ptbxl_test.append(file)
            if dataset == 'samitrop':
                self.sami_test.append(file)

    def build_unified_index(self):
        """
        Builds a unified list of (record_path, start_sample_index, label, dataset_name)
        """
        unified_index = []
        
        for dataset_name, config in self.datasets_config.items():
            print(f"Processing {dataset_name}...")
            dataset_index = self._build_dataset_index(dataset_name, config)
            unified_index.extend(dataset_index)
            
        return unified_index

    def _build_dataset_index(self, dataset_name, config):
        """Build index for a specific dataset"""
        dataset_path = config['path']
        dataset_type = config['type']
        original_fs = self.original_fs_dict[dataset_name]
        
        if dataset_type == 'code15':
            return self._build_code15_index(dataset_path, dataset_name, original_fs)
        elif dataset_type == 'ptbxl':
            return self._build_ptbxl_index(dataset_path, dataset_name, original_fs)
        elif dataset_type == 'sami':
            return self._build_sami_index(dataset_path, dataset_name, original_fs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _build_code15_index(self, db_path, dataset_name, original_fs):
        """Build index for CODE-15 dataset"""
        with open('sig_len.pkl', 'rb') as f:
            signal_len_dict = pickle.load(f)

        index = []
        for file in self.code15_test:
            with open(file, 'rb') as f:
                record = pickle.load(f)
            label = int(record.comments[2].split()[2] == 'True')
            total_samples = record.p_signal.shape[0]

            window_samples = self.desired_signal_length * original_fs
            if label == -1:  # Negative: only 1 window starting at 0
                if total_samples // window_samples > 2:
                    max_start = total_samples - window_samples
                    start = random.randint(window_samples, max_start-window_samples)
                    index.append((file, start, label, dataset_name))
            else:  # Positive: add multiple windows
                n_windows = total_samples // window_samples
                for i in range(1, n_windows-1):
                    start = i * window_samples
                    index.append((file, start, label, dataset_name))

        return index

    def _build_ptbxl_index(self, db_path, dataset_name, original_fs):
        """Build index for PTB-XL dataset"""
        index = []
        window_samples = self.desired_signal_length * original_fs

        for file in self.ptbxl_test:
            # PTB-XL are all negatives, single window per record
            record = wfdb.rdrecord(file)
            total_samples = record.p_signal.shape[0]
            n_windows = total_samples // window_samples
            for i in range(1, n_windows-1):
                start = i * window_samples
                index.append((file, start, 0, dataset_name))

        return index

    def _build_sami_index(self, db_path, dataset_name, original_fs):
        """Build index for Sami-Trop dataset"""
        index = []
        window_samples = self.desired_signal_length * original_fs
        
        for file in self.sami_test:
            # Sami-Trop are all positives: add multiple windows like CODE-15 positives

            record = wfdb.rdrecord(file)
            total_samples = record.p_signal.shape[0]
            n_windows = total_samples // window_samples
            for i in range(1, n_windows-1):
                start = i * window_samples
                index.append((file, start, 1, dataset_name))

        return index

    def __getitem__(self, index):
        record_path, start_sample, label, dataset_name = self.record_info[index]
        original_fs = self.original_fs_dict[dataset_name]
        
        if dataset_name == 'code15':
            with open(record_path, 'rb') as f:
                record = pickle.load(f)
            signal_slice = record.p_signal[start_sample:start_sample + self.desired_signal_length * original_fs]
        else:  # ptbxl or sami
            record = wfdb.rdrecord(record_path)
            signal_slice = record.p_signal[start_sample:start_sample + self.desired_signal_length * original_fs]
        
        signal = self.transform(signal_slice, original_fs, self.new_fs, self.desired_signal_length)

        exam_id = -1 # -1 if sami sample
        if dataset_name == 'code15' or dataset_name == 'ptbxl':
            exam_id = int(record.record_name.split('_')[1])
        else:
            exam_id = int(record.record_name.split('.')[0])

        return signal, label, dataset_name_to_id[dataset_name], exam_id 

    def __len__(self):
        return len(self.record_info)

    def get_dataset_indices(self, dataset_name):
        """Get indices for a specific dataset"""
        return [i for i, (_, _, _, ds_name) in enumerate(self.record_info) if ds_name == dataset_name]

    def get_dataset_stats(self):
        """Get statistics for each dataset"""
        stats = {}
        for dataset_name in self.datasets_config.keys():
            indices = self.get_dataset_indices(dataset_name)
            labels = [self.record_info[i][2] for i in indices]
            stats[dataset_name] = {
                'total': len(labels),
                'positives': sum(labels),
                'negatives': len(labels) - sum(labels)
            }
        return stats


########## train-test Split
def train_test_split_per_dataset(dataset, batch_size, num_workers=0, seed=42):
    
    random.seed(seed)
    np.random.seed(seed)

    # Create DataLoaders
    indices = range(len(dataset.record_info))
    dl = DataLoader(dataset, batch_size=batch_size, 
                         sampler=SubsetRandomSampler(indices), 
                         num_workers=num_workers)
    
    return dl


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial training
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        lambda_ = grad_output.new_tensor(lambda_)
        dx = -lambda_ * grad_output
        return dx, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial domain adaptation
    """
    def __init__(self, input_dim, hidden_dim=512, num_domains=3, dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
    def forward(self, x):
        return self.domain_classifier(x)

class AdversarialResNet1D(nn.Module):
    """
    Modified ResNet1D with domain adaptation capabilities, supporting optional GRU and Attention.
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, 
                 n_classes, num_domains=3, downsample_gap=6, increasefilter_gap=12, 
                 use_bn=True, use_do=True, verbose=False,
                 with_attention=False, n_attention_heads=8, with_gru=False, n_gru_layers=1, gru_hidden_size=None): # Added GRU parameters
        super(AdversarialResNet1D, self).__init__()
        
        # Determine the dimension of features output by the ResNet's convolutional blocks
        # This is the 'out_channels' of the last residual block as calculated in ResNet1D's __init__
        # and passed as the input_size for GRU/Attention.
        resnet_output_channels = int(base_filters * 2**((n_block-1)//increasefilter_gap))

        # Determine the actual feature dimension after optional GRU, which will be the input for the classifiers
        feature_dim_for_classifiers = gru_hidden_size if with_gru and gru_hidden_size is not None else resnet_output_channels
        
        # Create the feature extractor (ResNet1D without final classification layer)
        self.feature_extractor = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            n_block=n_block,
            n_classes=n_classes,  # We'll replace the final layer
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_bn=use_bn,
            use_do=use_do,
            verbose=verbose,
            with_attention=with_attention, # Pass attention flag
            n_attention_heads=n_attention_heads, # Pass number of attention heads
            with_gru=with_gru, # Pass GRU flag
            n_gru_layers=n_gru_layers, # Pass number of GRU layers
            gru_hidden_size=gru_hidden_size # Pass GRU hidden size
        )
        
        # Replace the final dense layer for classification
        self.label_classifier = nn.Linear(feature_dim_for_classifiers, n_classes)
        
        # Add gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()
        
        # Add domain classifier
        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim_for_classifiers,
            num_domains=num_domains
        )
        
        # Remove the original dense layer from feature extractor, as it's replaced here
        self.feature_extractor.dense = nn.Identity()
        
    def forward(self, x, lambda_grl=1.0):
        # Extract features from the modified ResNet1D (which now includes optional GRU and Attention)
        # The output 'features' will be (batch_size, final_feature_dim) after global average pooling
        features = self.feature_extractor(x)
        
        # Label prediction
        label_pred = self.label_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = self.gradient_reversal(features)
        self.gradient_reversal.lambda_ = lambda_grl
        domain_pred = self.domain_classifier(reversed_features)
        
        return label_pred, domain_pred, features


########## Forward Epoch
# Modified forward_epoch function
def adversarial_forward_epoch(model, dl, label_loss_function, domain_loss_function, 
                            optimizer, train_mode=True, desc=None, device=torch.device('cpu'),
                            lambda_grl=1.0, domain_loss_weight=1.0):
    """
    Modified forward epoch for adversarial domain adaptation
    
    Args:
        model: AdversarialResNet1D model
        dl: DataLoader
        label_loss_function: Loss function for label classification (BCEWithLogitsLoss)
        domain_loss_function: Loss function for domain classification (CrossEntropyLoss)
        optimizer: Optimizer
        train_mode: Whether in training mode
        desc: Description for progress bar
        device: Device to run on
        lambda_grl: Lambda parameter for gradient reversal layer
        domain_loss_weight: Weight for domain loss
    """
    total_label_loss = 0
    total_domain_loss = 0
    total_loss = 0
    
    # Initialize lists to store data for t-SNE
    all_features = []
    all_labels = []
    all_domain_ids = []

    with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
        for i_batch, (X, y_label, y_domain, y_exams_id) in enumerate(dl):
            X = X.to(device)
            y_label = y_label.to(device)
            y_domain = y_domain.to(device)
            y_exams_id = y_exams_id.to(device)
            
            # Forward pass
            label_pred, domain_pred, features = model(X, lambda_grl=lambda_grl)
            label_pred = torch.squeeze(label_pred)
            
            # Label loss
            y_label_float = y_label.type(torch.float32)
            label_loss = label_loss_function(label_pred, y_label_float)
            
            # Domain loss
            domain_loss = domain_loss_function(domain_pred, y_domain)
            
            # Combined loss
            total_batch_loss = label_loss + domain_loss_weight * domain_loss
            
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += total_batch_loss.item()
            
            if train_mode:
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
            
            # Store predictions and true labels
            if i_batch == 0:
                y_true_epoch = y_label_float
                y_pred_epoch = label_pred
                y_domain_true_epoch = y_domain
                y_domain_pred_epoch = domain_pred
                
                y_datasets_id_2d = y_domain.unsqueeze(1)
                y_exams_id_2d = y_exams_id.unsqueeze(1)
                y_ids_epoch = torch.concat((y_datasets_id_2d, y_exams_id_2d), dim=1)
            else:
                y_true_epoch = torch.concat((y_true_epoch, y_label_float))
                y_pred_epoch = torch.concat((y_pred_epoch, label_pred))
                y_domain_true_epoch = torch.concat((y_domain_true_epoch, y_domain))
                y_domain_pred_epoch = torch.concat((y_domain_pred_epoch, domain_pred))
                
                y_datasets_id_2d = y_domain.unsqueeze(1)
                y_exams_id_2d = y_exams_id.unsqueeze(1)
                y_ids_2d = torch.concat((y_datasets_id_2d, y_exams_id_2d), dim=1)
                y_ids_epoch = torch.concat((y_ids_epoch, y_ids_2d))
            
            # Collect features, labels, and domain IDs for t-SNE if in evaluation mode
            if not train_mode:
                all_features.append(features.detach().cpu())
                all_labels.append(y_label.detach().cpu())
                all_domain_ids.append(y_domain.detach().cpu())
            
            pbar.update(1)

    # Average losses over batches (like in original code)
    avg_total_loss = total_loss / len(dl)
    avg_label_loss = total_label_loss / len(dl)
    avg_domain_loss = total_domain_loss / len(dl)

    # Concatenate collected tensors if in evaluation mode
    if not train_mode:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_domain_ids = torch.cat(all_domain_ids, dim=0)
        return (avg_total_loss, avg_label_loss, avg_domain_loss, 
                y_true_epoch, y_pred_epoch, y_domain_true_epoch, y_domain_pred_epoch, y_ids_epoch,
                all_features, all_labels, all_domain_ids) # Return features, labels, domains
    else:
        return (avg_total_loss, avg_label_loss, avg_domain_loss, 
                y_true_epoch, y_pred_epoch, y_domain_true_epoch, y_domain_pred_epoch, y_ids_epoch,
                None, None, None) # Return None for t-SNE related outputs if in train mode


# Lambda scheduler for gradient reversal layer
class LambdaScheduler:
    """
    Scheduler for lambda parameter in gradient reversal layer
    Implements the schedule from DANN paper: λ = 2/(1+exp(-10*progress)) - 1
    """
    def __init__(self, gamma=10.0):
        self.gamma = gamma
    
    def get_lambda(self, epoch, max_epochs):
        progress = epoch / max_epochs
        return 2.0 / (1.0 + np.exp(-self.gamma * progress)) - 1.0


def main():

    train_set = torch.load('train_set.pt', map_location=torch.device('cpu'))
    test_set = torch.load('test_set.pt', map_location=torch.device('cpu'))
    print("Original train set size: {0}".format(len(train_set)))
    print("Original test set size: {0}".format(len(test_set)))

    # Create unified dataset
    unified_dataset = UnifiedECGDataset(
        datasets_config=datasets_config,
        transform=ECGTransform(),
        original_fs_dict=original_fs_dict,
        new_fs=256,
        desired_signal_length=WINDOW_LENGTH,
        train_set=train_set
    )

    unified_dataset_test = UnifiedECGDatasetTestSet(
        datasets_config=datasets_config,
        transform=ECGTransform(),
        original_fs_dict=original_fs_dict,
        new_fs=256,
        desired_signal_length=WINDOW_LENGTH,
        test_set=test_set
    )

    print(f'Unified training dataset length: {len(unified_dataset)}')
    print('Dataset statistics:')
    stats = unified_dataset.get_dataset_stats()
    for name, stat in stats.items():
        print(f"{name}: {stat}")

    print(f'Unified test dataset length: {len(unified_dataset_test)}')
    print('Dataset statistics:')
    stats = unified_dataset_test.get_dataset_stats()
    for name, stat in stats.items():
        print(f"{name}: {stat}")


    dl_train = train_test_split_per_dataset(
        dataset=unified_dataset,
        batch_size=512,
        num_workers=8
    )

    dl_test = train_test_split_per_dataset(
        dataset=unified_dataset_test,
        batch_size=512,
        num_workers=8
    )

    print(f'Training set size: {len(dl_train.sampler.indices)}')
    print(f'Test set size: {len(dl_test.sampler.indices)}')


    ####### Defining the network
    # Create adversarial model instead of regular ResNet1D
    ecg_net = AdversarialResNet1D(
        in_channels=12, 
        base_filters=128, 
        kernel_size=11, 
        stride=2, 
        groups=1, 
        n_block=48, 
        n_classes=1,
        num_domains=3,  # code15, ptbxl, sami
        downsample_gap=6, 
        increasefilter_gap=12,
        with_attention=True, # Set this to True to enable attention
        n_attention_heads=8, # Specify the number of attention heads
        with_gru=True, # Set this to True to enable GRU layers
        n_gru_layers=2 # Specify the number of stacked GRU layers (e.g., 2 or 3)
    )

    ####### Finding positive samples in training set
    GPU_0 = torch.device(1)
    pos_number = 0
    negative_number = 0
    dataset_count = {0 : 0, 1 : 0, 2 : 0}
    for signals_train, labels_train, dataset_id, _ in dl_train:
        pos_number += labels_train.sum()
        negative_number += (labels_train==0).sum()
        dataset_count[0] += torch.sum(dataset_id == 0)
        dataset_count[1] += torch.sum(dataset_id == 1)
        dataset_count[2] += torch.sum(dataset_id == 2)

    print("In Train Set:")
    print("Number of positive samples: {0}".format(pos_number))
    print("Number of negative samples: {0}".format(negative_number))
    print("Datasets counts: {0}".format(dataset_count))

    total_count = sum(dataset_count.values())
    domain_weights = torch.tensor([total_count / dataset_count[0], total_count / dataset_count[1], total_count / dataset_count[2]])
    domain_weights = domain_weights.to(GPU_0)

    # Define loss functions
    pos_weight = negative_number / pos_number
    label_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    domain_loss_function = nn.CrossEntropyLoss(weight=domain_weights)
    print("Domain Weights: {0}".format(domain_weights))

    learning_rate = 0.0001 
    optimizer = torch.optim.Adam(params=ecg_net.parameters(), lr=learning_rate)

    # Lambda scheduler
    lambda_scheduler = LambdaScheduler()

        # Training loop
    epochs = 30
    domain_loss_weight = 0.8  # Adjust this hyperparameter

    # Loss tracking - EXPANDED
    train_loss_vec = []
    test_loss_vec = []
    train_label_loss_vec = []
    test_label_loss_vec = []
    train_domain_loss_vec = []
    test_domain_loss_vec = []
    
    GPU_0 = torch.device(1) # Using device 1 (second GPU if available, else will error if only 1 GPU and it's 0)
    print(GPU_0)

    ecg_net = ecg_net.to(GPU_0)

    # Initialize variables to store features and labels for t-SNE from the last epoch
    final_test_features = None
    final_test_labels = None
    final_test_domain_ids = None

    for i_epoch in range(epochs):
        print(f'Epoch: {i_epoch+1}/{epochs}')
        lambda_grl = lambda_scheduler.get_lambda(i_epoch, epochs)
        
        # Training
        ecg_net.train()
        (train_total_loss, train_label_loss, train_domain_loss, 
         y_true_train, y_pred_train, _, _, _, _, _, _) = adversarial_forward_epoch( # Added _ for t-SNE outputs in train mode
            ecg_net, dl_train, label_loss_function, domain_loss_function,
            optimizer, train_mode=True, desc='Train', device=GPU_0,
            lambda_grl=lambda_grl, domain_loss_weight=domain_loss_weight
        )
        # Testing
        ecg_net.eval()
        with torch.no_grad():
            (test_total_loss, test_label_loss, test_domain_loss,
             y_true_test, y_pred_test, domain_true_test, domain_pred_test, _, # Added _ for y_ids
             current_epoch_features, current_epoch_labels, current_epoch_domain_ids) = adversarial_forward_epoch( # Captured t-SNE outputs
                ecg_net, dl_test, label_loss_function, domain_loss_function,
                optimizer, train_mode=False, desc='Test', device=GPU_0,
                lambda_grl=lambda_grl, domain_loss_weight=domain_loss_weight
        )

        # Store metrics
        train_loss = train_label_loss + train_domain_loss
        test_loss = test_label_loss + test_domain_loss
        train_loss_vec.append(train_loss)
        test_loss_vec.append(test_loss)
        train_label_loss_vec.append(train_label_loss)
        test_label_loss_vec.append(test_label_loss)
        train_domain_loss_vec.append(train_domain_loss)
        test_domain_loss_vec.append(test_domain_loss)

        # Print epoch summary
        print(f'Train - Total: {train_loss:.4f}, Label: {train_label_loss:.4f}, Domain: {train_domain_loss:.4f}')
        print(f'Test  - Total: {test_loss:.4f}, Label: {test_label_loss:.4f}, Domain: {test_domain_loss:.4f}')

        # Save epoch results
        torch.save(y_pred_test, f'test_pred_epoch{i_epoch}.pt')
        torch.save(y_true_test, f'test_true_epoch{i_epoch}.pt')
        torch.save(domain_pred_test, f'test_domain_pred_epoch{i_epoch}.pt')
        torch.save(domain_true_test, f'test_domain_true_epoch{i_epoch}.pt')

        # For t-SNE, save features and true labels/domain IDs from the last epoch
        if i_epoch == epochs - 1:
            final_test_features = current_epoch_features
            final_test_labels = current_epoch_labels
            final_test_domain_ids = current_epoch_domain_ids

    # Save all loss vectors
    torch.save(train_loss_vec, 'train_loss_vec.pt')
    torch.save(test_loss_vec, 'test_loss_vec.pt')
    torch.save(train_label_loss_vec, 'train_label_loss_vec.pt')
    torch.save(test_label_loss_vec, 'test_label_loss_vec.pt')
    torch.save(train_domain_loss_vec, 'train_domain_loss_vec.pt')
    torch.save(test_domain_loss_vec, 'test_domain_loss_vec.pt')

    torch.save(ecg_net.state_dict(), 'model_weights.pth')
       
    # Final validation - save collected features and labels for t-SNE outside the loop
    # This ensures we get the features from the final state of the model.
    # Note: The existing final validation call in your original code only uses lambda_scheduler,
    # which is likely incorrect as it should be a scalar lambda_grl. I've updated it to use
    # the last computed lambda_grl.
    with torch.no_grad():
        (_, _, _, y_true, y_pred, domain_true, domain_pred, y_ids,
         final_test_features, final_test_labels, final_test_domain_ids) = adversarial_forward_epoch(
            ecg_net, dl_test, label_loss_function, domain_loss_function, optimizer,
            train_mode=False, desc='Final Test', device=GPU_0, lambda_grl=lambda_grl, domain_loss_weight=domain_loss_weight # Pass the scalar lambda_grl
        )
        torch.save(y_true, 'y_true.pt')
        torch.save(y_pred, 'y_pred.pt')
        torch.save(domain_true, 'domain_true.pt')
        torch.save(domain_pred, 'domain_pred.pt')
        torch.save(y_ids, 'y_ids.pt')

        # Save features, labels, and domain IDs specifically for t-SNE
        if final_test_features is not None:
            torch.save(final_test_features, 'tsne_features.pt')
            torch.save(final_test_labels, 'tsne_labels.pt')
            torch.save(final_test_domain_ids, 'tsne_domain_ids.pt')
            print("Saved features, labels, and domain IDs for t-SNE analysis.")
    
if __name__ == '__main__':
    main()