"""Preprocesses the raw EEG file: channel selection, re-reference, bandpass 
filter, epoching, frequency downsampling, multivariate noise normalization 
(MVNN) and reshaping the data to: EEG channels × EEG time points.
Then, the dream EEG data is saved.

Parameters 

----------
project_dir : str
	Directory of the project folder.
PSG : str
	Used subject.
sfreq : int
	Downsampling frequency.
"""

import argparse
from ZW_func import epoching, mvnn, save_prepr

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--PSG', default='010_Morning', type=str)
parser.add_argument('--sfreq', default=100, type=int)
args = parser.parse_args()

print('>>> Zhang & Wamsley dream EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Epoch and sort the data
# =============================================================================
data, ch_names, times = epoching(args)

# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
whitened_data = mvnn(data)
del data

# =============================================================================
# Save the preprocessed data
# =============================================================================
save_prepr(args, whitened_data, ch_names, times)