import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--feature_selection',default=False,type=bool)
args = parser.parse_args()

print(f'>>> Running all scripts on Zhang & Wamsley REM dreams <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Extract feature maps
print(f'>>> Extract Feature maps <<<')
os.system(f'python ../project_directory/github_code/03_dnn_fmaps/alexnet_ZW_REMs.py')
os.system(f'python ../project_directory/github_code/03_dnn_fmaps/pca.py --dataset ZW_REMs')

# Compute the correlation scores
print(f'>>> Compute correlation scores <<<')
if args.feature_select == False:
    os.system(f'python ../project_directory/github_code/04_validation_test/corr_ZW_REM.py')
else:
    os.system(f'python ../project_directory/github_code/04_validation_test/corr_ZW_REM_fs.py')

# Plot the RDMs
print(f'>>> Plot the RDMs <<<')
os.system(f'python ../project_directory/github_code/05_plots/RDMs_REM.py --feature_selection {args.feature_select}')