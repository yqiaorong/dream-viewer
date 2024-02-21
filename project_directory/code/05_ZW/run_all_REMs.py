import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_selection',default=False, type=bool)
parser.add_argument('--all_or_best', default='all', type=str)
args = parser.parse_args()

print(f'>>> Running all scripts on Zhang & Wamsley REM dreams <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Extract feature maps
print(f'>>> Extract Feature maps <<<')
os.system(f'python project_directory/code/02_dnn_fmaps/alexnet_ZW_REMs.py --all_or_best {args.all_or_best}')
os.system(f'python project_directory/code/02_dnn_fmaps/pca.py --dataset ZW_REMs_{args.all_or_best}')

# Compute the correlation scores
print(f'>>> Compute correlation scores <<<')
if args.feature_select == False:
    os.system(f'python project_directory/code/03_encoding/encoding_ZW_REM.py --all_or_best {args.all_or_best}')
else:
    os.system(f'python project_directory/code/03_encoding/emcoding_ZW_REM_fs.py --all_or_best {args.all_or_best}')

# Plot the RDMs
print(f'>>> Plot the RDMs <<<')
os.system(f'python project_directory/code/04_plots/plot_RDMs_REM.py --feature_selection {args.feature_select} --all_or_best {args.all_or_best}')

# Plot the PCA clusters
print(f'>>> Plot the PCA clusters <<<')
os.system(f'python project_directory/code/04_plots/pca_cluster.py --all_or_best {args.all_or_best}')