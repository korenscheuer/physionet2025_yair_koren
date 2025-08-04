"# physionet2025_yair_koren" 
The full repository is available at: https://github.com/korenscheuer/physionet2025_yair_koren

Preprocessing procedure:
1. Run prepare code scripts for each dataset as provided by the official Physionet 2025 challenge: prepare_code15_data.py, prepare_ptbxl_data.py, prepare_samitrop_data.py. These sciprts rely on 'helper_code.py' which was also available as part of the challenge. The scripts will convert the available ECG raw data files to wfdb .dat and .hea files. In addition, the scripts were modified to output signel length files which include dictionaries of record names and their respective signal length. This was done to improve file reading speed when checking for length conditions. Moreover, code15 .dat and .hea files were later converted to a dumped python pickle (.pkl) files to further increase file reading speed.

2. Run bsqi_filtering.py script to filter samples by a bSQI score. This script will output a list 'record_info.pt' containing windows of every sample that fulfilled the bSQI criterion.

3. Run train_test_split.py script to divide all samples to train and test sets. The script will output 'train_set.pt' and 'test_set.pt' which are lists that contain the file paths after splitting as described in the paper (code15 95%-5%, ptbxl 92%-8%, samitrop 60%-40% train-test splits).

Model Training:
4. Run physionet_train.py script to train the model on the training set as described in the paper. The sciprt will output many .pt files that contain relevant information for evalution as well as the model final weights. The model network archeticture is included in 'gru_attentioned_resnet1d.py'.

Evaluation:
5. Finally, The jupyter notebook file evaluation.ipynb contains code blocks to visually evaluate the model's performance.