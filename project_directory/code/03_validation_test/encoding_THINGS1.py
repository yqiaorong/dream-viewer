"""Train the encoding model on THINGS EEG2 dataset, and test the encoding model
on THINGS EEG1 dataset.

Parameters
----------
train_data_dir : str
	Directory of the training data folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
test_dataset : str
    Used test dataset ('THINGS_EEG1')
"""

import os
import argparse
import numpy as np
import pingouin as pg
from tqdm import tqdm
from matplotlib import pyplot as plt
from encoding_func import train_model_THINGS2, corr_THINGS1

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='THINGS_EEG1',type=str)
args = parser.parse_args()

print(f'>>> Test the encoding model on {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Train the encoding model 
# =============================================================================

reg = train_model_THINGS2(args)

# =============================================================================
# Predict the EEG data of THINGS EEG2 test images
# =============================================================================

### Load the test DNN feature maps ###
# Load the test DNN feature maps directory
dnn_parent_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                            'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                            args.dnn, 'pretrained-True', 'layers-all')
# Load the test DNN feature maps (images,3000)
dnn_fmaps_test = np.load(os.path.join(dnn_parent_dir, 'pca_feature_maps_test.npy'
                        ), allow_pickle=True).item()

### Predict the EEG test data using the encoding model ###
# Predict the test EEG data 
# THINGS_EEG1 (images, 17 x 100) 
pred_eeg = reg.predict(dnn_fmaps_test['all_layers'])
print('pred_eeg_data_test shape', pred_eeg.shape)

# =============================================================================
# Test the encoding model on THINGS1 each test subject
# =============================================================================

# Test subjects list
test_subjs = [x for x in range(1, 51) if x != 6]

# Get the encoding accuracy for each subject
tot_accuracy = np.empty((len(test_subjs),100))
for i, test_subj in enumerate(tqdm(test_subjs, desc='THINGS1 subjects')):
    accuracy, times = corr_THINGS1(args, pred_eeg, test_subj)
    tot_accuracy[i] = accuracy
        
# =============================================================================
# Plot the correlation results
# =============================================================================
    
# Create the saving directory
save_dir = os.path.join(args.project_dir, 'results', 
                        f'{args.test_dataset}_validation_test')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

### All subjects ###
plt.figure(1)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# Set the plot colour spectum
cmap = "cividis"
colours = plt.colormaps[cmap](np.linspace(0,1,len(test_subjs)))
# Plot
for i in range(len(test_subjs)):
    plt.plot(times, tot_accuracy[i], color = colours[i], alpha=0.2)
plt.plot(times, np.mean(tot_accuracy,0), color='k', label='Correlation scores')
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.3)
plt.title(f'Encoding accuracy on {args.test_dataset} ({args.dnn})')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, 
                         f'Encoding accuracy on {args.test_dataset} ({args.dnn}).jpg'))

### Average with confidence interval ###
# Set random seed for reproducible results
seed = 20200220
# Set the confidence interval
ci = np.empty((2,len(times)))
# Plot
plt.figure(2)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# Calculate the confidence interval
for i in range(len(times)):
    ci[:,i] = pg.compute_bootci(tot_accuracy[:,i], func='mean', seed=seed)
# Plot the results with confidence interval
plt.plot(times, np.mean(tot_accuracy,0), color='salmon', 
         label='correlation mean scores with 95 \% confidence interval')
plt.fill_between(times, np.mean(tot_accuracy,0), ci[0], color='salmon', alpha=0.2)
plt.fill_between(times, np.mean(tot_accuracy,0), ci[1], color='salmon', alpha=0.2)
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.05, top=.1)
plt.title(f'Averaged ncoding accuracy on {args.test_dataset} ({args.dnn})')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, 
                         f'Averaged encoding accuracy on {args.test_dataset} ({args.dnn}).jpg'))