import torch
import os
import random
import numpy as np

# Load record_info output file from bsqi_filtering.py
record_info = torch.load('record_info.pt', map_location=torch.device('cpu'))
print(record_info[:10])


########## Save file paths from all 3 datasets in a list
all_files_lst = []

db = 'code15/'
for directory in os.listdir(db):
    if not directory.endswith('.zip'):
        for file in os.listdir('{0}/{1}/{1}/'.format(db, directory)):
            all_files_lst.append('{0}{1}/{2}'.format(db, directory, file))

db = 'samitrop/'
for file in os.listdir(db):
    if file.endswith('.hea'):
        all_files_lst.append(db + file[:-4])

db = 'ptbxl_100hz_patient_id/'
for directory in os.listdir(db):
    for file in os.listdir(db + directory):
        if file.endswith('.hea'):
            all_files_lst.append('{0}{1}/{2}'.format(db, directory, file[:-4]))

val_file_lst = []

files_lst = [] # List of files that were bSQI filtered
for item in record_info:
    file_path = item[0]
    if file_path not in files_lst:
        files_lst.append(file_path)

for item in all_files_lst:
	# If a file is not in the bSQI filtered list, put it in a separate list for validation
    if item not in files_lst:
        val_file_lst.append(item)

torch.save(files_lst, 'high_bsqi_file_paths.pt')
torch.save(val_file_lst, 'low_bsqi_file_paths.pt')

########## train-test split
def divide_datasets(file_paths):
    code15_files = []
    ptbxl_files = []
    sami_files = []
    for file in file_paths:
        dataset = file.split('/')[0]
        if dataset == 'code15':
            code15_files.append(file)
        if dataset == 'ptbxl_100hz_patient_id':
            ptbxl_files.append(file)
        if dataset == 'samitrop':
            ptbxl_files.append(file)

    return code15_files, ptbxl_files, sami_files

def train_test_split_per_dataset(file_paths, dataset_splits, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    train_indices = []
    test_indices = []

    code15_files, ptbxl_files, sami_files = divide_datasets(file_paths)

    for dataset_name, train_ratio in dataset_splits.items():
        if dataset_name == 'code15':
            # For CODE-15, split by patients to avoid leakage
            patients = {}
            for full_file_path in code15_files:
                file = os.path.basename(full_file_path)
                patient = file.split('_')[0]
                if patient not in patients:
                    patients[patient] = []
                patients[patient].append(full_file_path)
            
            # Shuffle and split patients
            patient_ids = list(patients.keys())
            random.shuffle(patient_ids)
            
            total_windows = len(code15_files)
            train_count = 0
            
            for patient in patient_ids:
                if train_count < total_windows * train_ratio:
                    train_indices.extend(patients[patient])
                    train_count += len(patients[patient])
                else:
                    test_indices.extend(patients[patient])
        elif dataset_name == 'ptbxl':
            # For PTB-XL, split by patients to avoid leakage (same as code15)
            patients = {}
            for full_file_path in ptbxl_files:
                file = os.path.basename(full_file_path)
                patient = file.split('_')[0]
                if patient not in patients:
                    patients[patient] = []
                patients[patient].append(full_file_path)
            
            # Shuffle and split patients
            patient_ids = list(patients.keys())
            random.shuffle(patient_ids)
            
            total_windows = len(ptbxl_files)
            train_count = 0
            
            for patient in patient_ids:
                if train_count < total_windows * train_ratio:
                    train_indices.extend(patients[patient])
                    train_count += len(patients[patient])
                else:
                    test_indices.extend(patients[patient])
        else:
            # For Sami-Trop, simple random split (assuming different patients)
            dataset_indices = range(len(sami_files))
            random.shuffle(dataset_indices)
            split_point = int(len(dataset_indices) * train_ratio)
            train_indices.extend(dataset_indices[:split_point])
            test_indices.extend(dataset_indices[split_point:])

    return train_indices, test_indices


high_bsqi_files = torch.load('high_bsqi_file_paths.pt', map_location=torch.device('cpu'))
low_bsqi_files = torch.load('low_bsqi_file_paths.pt', map_location=torch.device('cpu'))

test_lst = []
high_bsqi_stratify = []
low_bsqi_patients = {}
for file in low_bsqi_files:
    record_name = file.split('/')[-1]
    dataset = file.split('/')[0]
    patient = record_name.split('_')[0]
    if dataset not in low_bsqi_patients:
        low_bsqi_patients[dataset] = []

    low_bsqi_patients[dataset].append(patient)
    test_lst.append(file)

for file in high_bsqi_files:
    record_name = file.split('/')[-1]
    patient = record_name.split('_')[0]
    dataset = file.split('/')[0]
    if patient in low_bsqi_patients[dataset]:
        test_lst.append(file)
    else:
        high_bsqi_stratify.append(file)

# Define different train ratios for each dataset
dataset_splits = {
    'code15': 0.95,
    'ptbxl': 0.92,
    'sami': 0.6
}

train_lst, test_split_lst = train_test_split_per_dataset(
    file_paths=high_bsqi_stratify,
    dataset_splits=dataset_splits,
)

test_lst.extend(test_split_lst)

print(f'Training set size: {len(train_lst)}')
print(f'Test set size: {len(test_lst)}')

# Save output as train_set and test_set lists that contain the relevant file names in each set
torch.save(train_lst, 'train_set.pt')
torch.save(test_lst, 'test_set.pt')