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

from scipy.signal import butter, filtfilt, find_peaks

def pan_tompkins_qrs_detector(ecg_signal: np.ndarray, fs: int) -> np.ndarray:
    """
    A simplified Pan-Tompkins-like QRS detector.

    Args:
        ecg_signal (np.ndarray): The input ECG signal.
        fs (int): The sampling frequency of the ECG signal in Hz.

    Returns:
        np.ndarray: An array of detected R-peak indices.
    """

    # --- Step 1: Band-Pass Filtering (5-15 Hz) ---
    # Design a Butterworth band-pass filter
    lowcut = 5.0
    highcut = 15.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band') # Using a 1st order filter for simplicity
    filtered_ecg = filtfilt(b, a, ecg_signal)

    # --- Step 2: Differentiation ---
    # Uses a 5-point derivative approximation for smoothing
    # y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / (1/8 * fs)
    # Source: Pan and Tompkins (1985)
    differentiated_ecg = np.diff(filtered_ecg, n=1) # Simple difference for differentiation

    # --- Step 3: Squaring ---
    squared_ecg = differentiated_ecg ** 2

    # --- Step 4: Moving Window Integration ---
    window_size = int(0.150 * fs) # 150 ms window for integration (common for QRS)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')

    # --- Step 5: Adaptive Thresholding and Peak Detection ---
    # This is a highly simplified adaptive threshold.
    # A more robust implementation would involve dynamic thresholds for signal and noise.

    # Find peaks in the integrated signal
    # Properties like 'distance' and 'height' are crucial for accurate detection.
    # 'distance': Minimum number of samples between peaks (refractory period, ~200ms)
    # 'height': Minimum amplitude to be considered a peak (adaptive).
    
    # Initial estimate for peak height
    peak_height_threshold = np.mean(integrated_ecg) * 0.5 # A simple heuristic

    # Refractory period: no two R-peaks should be closer than 200 ms
    refractory_samples = int(0.200 * fs)

    # Find peaks
    r_peaks_indices, properties = find_peaks(integrated_ecg,
                                             height=peak_height_threshold,
                                             distance=refractory_samples)

    # Optional: Refine peak detection by going back to the original filtered ECG
    # and finding the local maximum around the detected integrated peak.
    refined_r_peaks = []
    search_window = int(0.050 * fs) # Search 50 ms around the detected peak

    for peak_idx in r_peaks_indices:
        start_idx = max(0, peak_idx - search_window)
        end_idx = min(len(filtered_ecg), peak_idx + search_window)
        
        # Find the maximum within the window in the original filtered ECG
        window = filtered_ecg[start_idx:end_idx]
        if window.size > 0:
            local_max_offset = np.argmax(np.abs(window)) # Use abs to handle inverted QRS
            refined_r_peaks.append(start_idx + local_max_offset)
        
    return np.array(refined_r_peaks, dtype=int)

def calculate_bsqi(signal: np.ndarray, fs: int, tolerance_ms: int = 150) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculates bSQI using two custom (simplified) QRS detectors.
    """
    # Detector 1: Pan-Tompkins
    xqrs_inds = pan_tompkins_qrs_detector(signal, fs)

    # Detector 2: A slightly modified Pan-Tompkins or another simple detector
    # For a meaningful bSQI, these two should ideally be different algorithms
    # or significantly different parameterizations of the same algorithm.
    # Here, a slight variation in sampling frequency passed to simulate a different detector.
    # In a real scenario, you might have different filter parameters or peak detection logic.
    gqrs_inds = pan_tompkins_qrs_detector(signal, fs * 0.95) # Example of a slight variation

    # Handle cases where no beats are detected by one or both algorithms
    if xqrs_inds.size == 0 and gqrs_inds.size == 0:
        return 1.0, xqrs_inds, gqrs_inds # Both detected nothing, consider it perfect agreement for no beats
    
    if xqrs_inds.size == 0 or gqrs_inds.size == 0:
        return 0.0, xqrs_inds, gqrs_inds # One detected beats, other didn't. Bad agreement.

    tolerance_samples = int(tolerance_ms * fs / 1000)
    
    true_positives = 0
    xqrs_pointer = 0
    gqrs_pointer = 0

    # This loop assumes both xqrs_inds and gqrs_inds are sorted
    while xqrs_pointer < len(xqrs_inds) and gqrs_pointer < len(gqrs_inds):
        xqrs_beat = xqrs_inds[xqrs_pointer]
        gqrs_beat = gqrs_inds[gqrs_pointer]

        if abs(xqrs_beat - gqrs_beat) <= tolerance_samples:
            true_positives += 1
            xqrs_pointer += 1
            gqrs_pointer += 1
        elif gqrs_beat < xqrs_beat:
            gqrs_pointer += 1
        else: # xqrs_beat < gqrs_beat
            xqrs_pointer += 1
            
    total_beats = len(xqrs_inds) + len(gqrs_inds)
    
    # Avoid division by zero if both are empty (already handled)
    if total_beats == 0:
        bsqi_score = 1.0 # Should be handled by the above if condition
    else:
        bsqi_score = 2 * true_positives / total_beats
    
    return bsqi_score, xqrs_inds, gqrs_inds

def bSQI_treshold(signal, fs, score_threshold, percent_threshold):
    bSQI_arr = np.zeros(12)

    for idx, lead in enumerate(signal):
        bSQI, _, _ = calculate_bsqi(lead, fs)
        bSQI_arr[idx] = bSQI

    count_above_threshold = np.sum(bSQI_arr > score_threshold)
    if count_above_threshold / len(bSQI_arr) > percent_threshold:
        return True
    return False


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
    def __init__(self, datasets_config, transform, original_fs_dict, new_fs, desired_signal_length):
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
        self.record_info = self.build_unified_index()

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
        for directory in os.listdir(db_path):
            if directory.endswith('.zip'):
                continue
            folder_path = os.path.join(db_path, directory)
            fake_folder_path = os.path.join(folder_path, directory)
            print(directory)
            for file in os.listdir(fake_folder_path):
                if not file.endswith('.pkl'):
                    continue
                path = os.path.join(fake_folder_path, file)
                real_path = os.path.join(folder_path, file)
                with open(path, 'rb') as f:
                    record = pickle.load(f)
                label = int(record.comments[2].split()[2] == 'True')
                total_samples = record.p_signal.shape[0]

                # If 75% of the leads are not above bSQI 91% then continue to next file
                if not bSQI_treshold(record.p_signal.T, original_fs_dict['code15'], 0.91, 0.75):
                    continue

                window_samples = self.desired_signal_length * original_fs
                if label == -1:  # Generate windows for both negative and positive samples
                    if total_samples >= window_samples:
                        max_start = total_samples - window_samples
                        start = random.randint(0, max_start)
                        index.append((path, start, label, dataset_name))
                else:  # Positive: add multiple windows
                    n_windows = total_samples // window_samples
                    for i in range(1, n_windows):
                        start = i * window_samples

                        real_path = db_path + directory + '/' + file
                        index.append((real_path, start, label, dataset_name))

        return index

    def _build_ptbxl_index(self, db_path, dataset_name, original_fs):
        """Build index for PTB-XL dataset"""
        with open('sig_len_ptb.pkl', 'rb') as f:
            signal_len_dict = pickle.load(f)
        
        index = []
        window_samples = self.desired_signal_length * original_fs

        for directory in os.listdir(db_path):
            folder_path = os.path.join(db_path, directory)
            for file in os.listdir(folder_path):
                current_signal_len = signal_len_dict['_'.join(file[:-4].split('_')[1:])]
                if file.endswith('.hea') and current_signal_len >= original_fs * self.desired_signal_length:
                    record_path = os.path.join(folder_path, file[:-4])
                    # PTB-XL are all negatives, single window per record

                    record = wfdb.rdrecord(record_path)
                    # If 75% of the leads are not above bSQI 91% then continue to next file
                    if not bSQI_treshold(record.p_signal.T, original_fs_dict['ptbxl'], 0.91, 0.75):
                        continue

                    total_samples = current_signal_len
                    n_windows = total_samples // window_samples
                    for i in range(1, n_windows):
                        start = i * window_samples
                        record_path = db_path + directory + '/' + file[:-4]
                        index.append((record_path, start, 0, dataset_name))

        return index

    def _build_sami_index(self, db_path, dataset_name, original_fs):
        """Build index for Sami-Trop dataset"""
        with open('sig_len_sami.pkl', 'rb') as f:
            signal_len_dict = pickle.load(f)
        
        index = []
        window_samples = self.desired_signal_length * original_fs
        
        for file in os.listdir(db_path):
            if file.endswith('.hea') and signal_len_dict[file[:-4]] >= original_fs * self.desired_signal_length:
                record_path = os.path.join(db_path, file[:-4])
                # Sami-Trop are all positives: add multiple windows like CODE-15 positives

                record = wfdb.rdrecord(record_path)
                # If 75% of the leads are not above bSQI 91% then continue to next file
                if not bSQI_treshold(record.p_signal.T, original_fs_dict['sami'], 0.91, 0.75):
                    continue
                total_samples = signal_len_dict[file[:-4]]
                n_windows = total_samples // window_samples
                for i in range(1, n_windows):
                    start = i * window_samples
                    record_path = db_path + file[:-4]
                    index.append((record_path, start, 1, dataset_name))

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

def main():
    # Create unified dataset
    unified_dataset = UnifiedECGDataset(
        datasets_config=datasets_config,
        transform=ECGTransform(),
        original_fs_dict=original_fs_dict,
        new_fs=256,
        desired_signal_length=WINDOW_LENGTH
    )

    # Save the record_info class variable to later be split to train-test
    torch.save(unified_dataset.record_info, 'record_info.pt')
    
if __name__ == '__main__':
    main()