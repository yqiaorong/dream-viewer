"""Test the THINGS2 decoding model on Zhang & Wamsley REM dreams.

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
from encoding_func import train_model_THINGS2, corr_ZW_spatial


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
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
                                                  
reg = train_model_THINGS2(args)


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
    for i in tqdm(range(REM_pred_eeg.shape[0]), desc=f'{dreams_eegs_names} correlation'):
        s, m = corr_ZW_spatial(args, REM_pred_eeg, e, i, crop_t)
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
    save_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation',
                            'REM_correlation_scores_s')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    file_name = item[6:]
    np.save(os.path.join(save_dir, file_name), results)

    del corr, mean_corr, results