"""Test the THINGS2 decoding model on Zhang & Wamsley

Parameters
----------
project_dir : str
	Directory of the training data folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
test_dataset : str
    Used test dataset ('Zhang_Wamsley')
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from corr_func import corr_s


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
args = parser.parse_args()

print(f'>>> Test the encoding model on Zhang & Wamsley <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Train the encoding model
# =============================================================================

### Load the training DNN feature maps ###
# Load the training DNN feature maps directory
dnn_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                args.dnn, 'pretrained-True', 'layers-all')
# Load the training DNN feature maps (16540, 3000)
dnn_fmaps_train = np.load(os.path.join(dnn_train_dir, 'pca_feature_maps_training.npy'), 
                            allow_pickle=True).item()

### Load the training EEG data ###
# Load the THINGS2 training EEG data directory
eeg_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                'THINGS_EEG2','preprocessed_data')
# Iterate over THINGS2 subjects
eeg_data_train = []
for train_subj in tqdm(range(1,11), desc='THINGS2 subjects'):
    # Load the THINGS2 training EEG data
    data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                  'preprocessed_eeg_training.npy'), allow_pickle=True).item()
    # Get the THINGS2 training channels and times
    if train_subj == 1:
        train_ch_names = data['ch_names']
    else:
        pass
    # Average the training EEG data across repetitions (16540,17,100)
    data = np.mean(data['preprocessed_eeg_data'], 1)
    # Crop the training EEG data between 0.1 and 0.25s (16540,17,15)
    data = data[:,:,30:45]
    # Average the training EEG data across time (16540,17)
    data = np.mean(data,axis=2)
    # Remove the EEG data from 'POz' channel (16540,16)
    POz_idx = train_ch_names.index('POz')
    data = np.delete(data,POz_idx,axis=1)
    # Append individual data
    eeg_data_train.append(data)
    del data
# Average the training EEG data across subjects : (16540,16)
eeg_data_train = np.mean(eeg_data_train,0)
# Delete unused channel names
del train_ch_names

### Train the encoding model ###
# Feature selection
reg = LinearRegression().fit(dnn_fmaps_train['all_layers'], eeg_data_train)
print(dnn_fmaps_train['all_layers'].shape)   
# Train the encoding models                                                      


# =============================================================================
# Predict the EEG data of dreams
# =============================================================================

### Load the test dream DNN feature maps ###
ZW_rem_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                            args.test_dataset, 'REMs')

# Load the test DNN feature maps directory
dnn_test_dir = os.path.join(ZW_rem_dir, 'dnn_feature_maps', 
                            'pca_feature_maps', args.dnn, 'pretrained-True', 
                            'layers-all')
dnn_test_list = os.listdir(dnn_test_dir)
# Load the test DNN feature maps 
REM_pred_eeg = []
for fmap in dnn_test_list:
    dnn_fmaps_test = np.load(os.path.join(dnn_test_dir,fmap,
                            ), allow_pickle=True).item()
    # Predict the EEG test data using the encoding model 
    pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
    REM_pred_eeg.append(pred_eeg_data_test)
    del pred_eeg_data_test
REM_pred_eeg = np.concatenate(REM_pred_eeg, axis=0)
print('pred eeg shape: ', REM_pred_eeg.shape)


# =============================================================================
# Compute the correlation scores for one dream
# =============================================================================	

# Test dreams EEG list
ZW_EEG_dir = os.path.join(ZW_rem_dir, 'preprocessed_data')
dreams_eegs_names = os.listdir(ZW_EEG_dir)

# Test dreams image list
ZW_img_dir = os.path.join(ZW_rem_dir, 'images')
ZW_rem_list = os.listdir(ZW_img_dir)

count = 0
single_dream_imgs_idx = [count]
for rem in ZW_rem_list:
    if rem.endswith('txt'):
        pass
    else:
        ZW_img_list = os.listdir(os.path.join(ZW_img_dir, rem))
        count += len(ZW_img_list)
        single_dream_imgs_idx.append(count)

for e, item in enumerate(dreams_eegs_names):
    # Set the cropped time points
    crop_t = 1000
    # The time points
    times = np.linspace(-int(crop_t/100), 0, crop_t)

    # The correlation scores (all images, times)
    corr = []
    # The correlation scores (all images)
    mean_corr = []

    # Iterate over images
    for i in tqdm(range(REM_pred_eeg.shape[0]), desc=f'correlation dream'):
        s, m = corr_s(args, REM_pred_eeg, e, i, crop_t)
        corr.append(s)
        mean_corr.append(m)
    corr = np.array(corr)
    mean_corr = np.array(mean_corr)

    # Save the all correlation results of one dream to the dictionary
    results = {}
    results['corresponding_img_idx'] = list(range(single_dream_imgs_idx[e],single_dream_imgs_idx[int(e+1)]))
    results['correlations'] = corr
    results['mean_correlations'] = mean_corr
    results['times'] = times

    # Create the saving directory
    save_dir = os.path.join(ZW_rem_dir, 'results', 'correlation_scores_s')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    file_name = item[6:]
    np.save(os.path.join(save_dir, file_name), results)

    del corr, mean_corr, results