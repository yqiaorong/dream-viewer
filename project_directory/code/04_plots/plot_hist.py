import os
import argparse
import numpy as np
from tqdm import tqdm
from plots_func import hist


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--percentile',default=0.95, type=float)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Plot the distribution of r scores of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Plot the histograms and the significance tests
# =============================================================================

# Load the directory
scores_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation', 
						'correlation_scores_'+args.st)
scores_list = os.listdir(scores_dir)

# Plots and significant results
sf = {}
sf['percentile'] = args.percentile
for item in tqdm(scores_list, desc='dreams'):
    s = hist(args, scores_dir, item)
    sf[item[:-4]] = s
    del s
    
# Save the significant results
save_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation', 
							'correlation_plots_'+args.st)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, str(args.percentile)+'_significant_r_scores'), sf)