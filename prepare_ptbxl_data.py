#!/usr/bin/env python

# Load libraries.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys
import wfdb

from helper_code import find_records, get_signal_files, is_integer

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True) # records100 or records500
    parser.add_argument('-d', '--ptbxl_database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-f', '--signal_format', type=str, required=False, default='dat', choices=['dat', 'mat'])
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Suppress stdout for noisy commands.
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout

# Convert .dat files to .mat files (optional).
def convert_dat_to_mat(record, write_dir=None):
    import wfdb.io.convert

    # Change the current working directory; wfdb.io.convert.matlab.wfdb_to_matlab places files in the current working directory.
    cwd = os.getcwd()
    if write_dir:
        os.chdir(write_dir)

    # Convert the .dat file to a .mat file.
    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_mat(record)

    # Remove the .dat file.
    os.remove(record + '.hea')
    os.remove(record + '.dat')

    # Rename the .mat file.
    os.rename(record + 'm' + '.hea', record + '.hea')
    os.rename(record + 'm' + '.mat', record + '.mat')

    # Update the header file with the renamed record and .mat file.
    with open(record + '.hea', 'r') as f:
        output_string = ''
        for l in f:
            if l.startswith('#Creator') or l.startswith('#Source'):
                pass
            else:
                l = l.replace(record + 'm', record)
                output_string += l

    with open(record + '.hea', 'w') as f:
        f.write(output_string)

    # Change the current working directory back to the previous current working directory.
    if write_dir:
        os.chdir(cwd)

# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)

# Run script.
def run(args):
    # Load the demographic information.
    df = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')

    # Identify the header files.
    records = find_records(args.input_folder)

    # Update the header files to include demographics data and copy the signal files unchanged.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        # Get patient_id for new naming format
        patient_id = str(int(row['patient_id']))

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = int(age) if is_integer(age) else float(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        height = row['height']
        height = int(height) if is_integer(height) else float(height)

        weight = row['weight']
        weight = int(weight) if is_integer(weight) else float(weight)

        # Assume that all of the patients are negative for Chagas, which is likely to be the case for every or almost every patient
        # in the PTB-XL dataset.
        label = False

        # Specify the label.
        source = 'PTB-XL'

        # Create new record name with patient_id prefix
        # Extract the suffix after the original ecg_id
        original_parts = record_basename.split('_')
        if len(original_parts) > 1:
            suffix = '_'.join(original_parts)
            new_record_basename = f"{patient_id}_{suffix}"
        else:
            new_record_basename = patient_id
        
        new_record = os.path.join(record_path, new_record_basename) if record_path else new_record_basename

        # Read the original record using wfdb
        input_record_path = os.path.join(args.input_folder, record)
        
        # Read the record data
        wfdb_record = wfdb.rdrecord(input_record_path)
        
        # Create output directory structure
        output_path = os.path.join(args.output_folder, record_path) if record_path else args.output_folder
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare the output record path
        output_record_path = os.path.join(args.output_folder, new_record)
        
        # Update record name in the wfdb record object
        wfdb_record.record_name = new_record_basename
        
        # Add demographic information to comments
        demographic_comments = [
            f'Age: {age}',
            f'Sex: {sex}', 
            f'Height: {height}',
            f'Weight: {weight}',
            f'Chagas label: {label}',
            f'Source: {source}'
        ]
        
        # Preserve existing comments (excluding demographic ones we're replacing)
        existing_comments = wfdb_record.comments if wfdb_record.comments else []
        filtered_comments = [c for c in existing_comments 
                           if not any(c.startswith(prefix) for prefix in 
                                    ['Age:', 'Sex:', 'Height:', 'Weight:', 'Chagas label:', 'Source:'])]
        
        # Combine comments
        wfdb_record.comments = filtered_comments + demographic_comments
        
        # Update base date and time
        wfdb_record.base_date = [int(dd), int(mm), int(yyyy)]
        wfdb_record.base_time = time_string


        record_output_name = output_record_path.split('\\')[-1]
        write_dir_path = os.path.dirname(output_record_path)

        # Write the record with new name using wfdb
        wfdb.wrsamp(
            record_name=record_output_name,
            fs=wfdb_record.fs,
            units=wfdb_record.units,
            sig_name=wfdb_record.sig_name,
            p_signal=wfdb_record.p_signal,
            comments=wfdb_record.comments,
            write_dir=write_dir_path,
            fmt=wfdb_record.fmt,
            adc_gain=wfdb_record.adc_gain,
            d_signal=wfdb_record.d_signal,
            baseline=wfdb_record.baseline
        )

        # Convert data from .dat files to .mat files as requested.
        if args.signal_format in ('mat', '.mat'):
            convert_dat_to_mat(output_record_path, write_dir=args.output_folder)

        # Recompute the checksums as needed.
        fix_checksums(output_record_path)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))