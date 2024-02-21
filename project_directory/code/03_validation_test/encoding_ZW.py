"""Train the encoding model on THINGS EEG2 dataset, and test the encoding model
on Zhang & Wamsley dream EEG dataset.

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
from tqdm import tqdm
from encoding_func import train_model_THINGS2, corr_ZW_spatial

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--start_dream_idx',default=146,type=int)
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
# List the folder of preprocessed ZW EEG data
dreams_eegs = os.listdir(ZW_EEG_dir)
dreams_eegs = [s[6:-4].replace('_', '') for s in dreams_eegs]
print('The total number of dreams in ZW: ', len(dreams_eegs))

# Test dreams images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
# List the folder of ZW images
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:].split('_')[0] for s in dreams_imgs]

# The list of indices of dreams with feature maps
dreams_eegs_idx = [idx for idx, item in enumerate(dreams_eegs) if item in 
                   dreams_imgs]
print('The total number of dreams with feature maps in ZW: ', len(dreams_eegs_idx))


# =============================================================================
# Train the encoding model
# =============================================================================

reg = train_model_THINGS2(args)


# =============================================================================
# Predict the EEG data of dreams
# =============================================================================

### Load the test dream DNN feature maps ###
# Load the test DNN feature maps directory
dnn_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                            args.test_dataset, 'dnn_feature_maps', 'pca_feature_maps', 
                            args.dnn, 'pretrained-True', 'layers-all')
# Load the test DNN feature maps (images, 3000)
dnn_fmaps_test = np.load(os.path.join(dnn_test_dir,'pca_feature_maps_dreams.npy'
                        ), allow_pickle=True).item()

### Predict the EEG test data using the encoding model ###
# Predict the test EEG data : (images,16)
pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
print('pred_eeg_data_test shape: ', pred_eeg_data_test.shape)


# =============================================================================
# Compute the correlation scores for one dream
# =============================================================================	

# Iterate over dreams
for dream_idx in range(args.start_dream_idx, 147):
    print('The current dream: ', dreams_eegs[dreams_eegs_idx[dream_idx]])
    # The list of indices of dream images of target dream
    dreams_imgs_idx = [idx for idx, item in enumerate(dreams_imgs) 
                    if item == dreams_eegs[dreams_eegs_idx[dream_idx]]]
    print(f'The total number of images for dream {dreams_eegs[dreams_eegs_idx[dream_idx]]}: ',
        len(dreams_imgs_idx), ' and the indice are: ', dreams_imgs_idx)

    # Set the cropped time points
    crop_t = 1000
    # The time points
    times = np.linspace(-int(crop_t/100), 0, crop_t)

    # The correlation scores (all images, times)
    corr = []
    # The correlation scores (all images)
    mean_corr = []

    # Iterate over images
    for i in tqdm(range(len(dreams_imgs)), desc='correlation'):
        s, m = corr_ZW_spatial(args, pred_eeg_data_test, 
                                                 dreams_eegs_idx[dream_idx], i, crop_t)
        corr.append(s)
        mean_corr.append(m)
    corr = np.array(corr)
    mean_corr = np.array(mean_corr)

    # Save the all correlation results of one dream to the dictionary
    results = {}
    results['corresponding_img_idx'] = dreams_imgs_idx
    results['correlations'] = corr
    results['mean_correlations'] = mean_corr
    results['times'] = times

    # Create the saving directory
    save_dir = os.path.join(args.project_dir,'results', 
                            f'{args.test_dataset}_correlation',
                            'correlation_scores_s')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    file_name = dreams_eegs[dreams_eegs_idx[dream_idx]]
    np.save(os.path.join(save_dir, file_name), results)

    del corr, mean_corr, dreams_imgs_idx, results