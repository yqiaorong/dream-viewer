"""Test the encoding model which is trained and tested on the same dream 
dataset: Zhang & Wamsley.

Parameters
----------
train_data_dir : str
	Directory of the training data folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
test_dataset : str
    Used test dataset ('Zhang_Wamsley')
"""

import os
import argparse
import numpy as np
import pingouin as pg
from tqdm import tqdm
from matplotlib import pyplot as plt

from encoding_func import model_ZW


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
args = parser.parse_args()

print(f'>>> Test the encoding model on {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
    
# =============================================================================
# Match the EEG data with images
# =============================================================================

# Test dreams EEG list
ZW_EEG_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'preprocessed_data')
dreams_eegs = os.listdir(ZW_EEG_dir)
dreams_eegs = [s[6:-4].replace('_', '') for s in dreams_eegs]

# Test dreams images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:].split('_')[0] for s in dreams_imgs]

# The list of indices of dreams with feature maps
dreams_eegs_idx = [idx for idx, item in enumerate(dreams_eegs) if item in 
                   dreams_imgs]
print('The total number of dreams with feature maps: ', len(dreams_eegs_idx))
print(dreams_eegs_idx)

# The list of indices of dream images of dreams
dreams_imgs_idx = []
for idx in dreams_eegs_idx:
	dreams_imgs_idx.append([i for i, item in enumerate(dreams_imgs) 
						 if item == dreams_eegs[idx]])
print('The number of dreams in img list:', len(dreams_imgs_idx))


# =============================================================================
# Train the encoding model, predict the EEG test data, and test the model
# =============================================================================
t = 2000
scores = model_ZW(args, dreams_eegs_idx, dreams_imgs_idx, t)
mean_scores = np.mean(scores, 0)


# =============================================================================
# Plot the correlation results
# =============================================================================
times = np.linspace(-int(t/100),0,t)

plt.figure(figsize=(10,4))
plt.plot([-int(t/100), 0], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
plt.plot(times, mean_scores, color = 'salmon', label='Correlation mean score')
plt.xlabel('Time (s)')
plt.xlim(left=-int(t/100), right=0)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.1)
plt.title(f'Encoding accuracy on Zhang & Wamsley (alexnet)')
plt.legend(loc='best')
plt.show()