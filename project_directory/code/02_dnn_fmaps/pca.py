"""PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.

Parameters
----------
project_dir : str
	Directory of the project folder.
dataset : str
    Used dataset (THINGS_EEG2), ('SCIP'), ('Zhang_Wamsley') or ('ZW_REMs').
dnn : str
	Used DNN among 'alexnet'.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	Whether to use 'all' or 'single' layers.
n_components : int
	Number of DNN feature maps PCA components retained.
"""

import os
import argparse
from pca_func import train_scaler_pca, apply_scaler_pca


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--dataset', default='Zhang_Wamsley', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=3000, type=int)
args = parser.parse_args()

print(f'>>> Apply PCA on the {args.dataset} feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Train the StandardScaler and PCA models on THINGS EEG2 training feature maps
# =============================================================================
# The standardization and PCA statistics computed on the THINGS training images.
scaler, pca, all_layers, layer_names = train_scaler_pca(args)


# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
if args.dataset == 'THINGS_EEG2':
    apply_scaler_pca(args, 'test', scaler, pca)
if args.dataset == 'SCIP':
    img_categories = ['cartoonflowers', 'cartoonguitar', 'cartoonpenguins']
    for img_category in img_categories:
        apply_scaler_pca(args, img_category, scaler, pca)
if args.dataset == 'Zhang_Wamsley':
    apply_scaler_pca(args, 'dreams', scaler, pca)
if args.dataset == 'ZW_REMs':
    dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'Zhang_Wamsley',
                    'REMs','dnn_feature_maps', 'full_feature_maps',args.dnn,
                    'pretrained-'+str(args.pretrained))
    img_categories = os.listdir(dir)
    for img_category in img_categories:
        apply_scaler_pca(args, img_category, scaler, pca)