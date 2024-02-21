import os
import argparse
import numpy as np
from tqdm import tqdm
from plots_func import plot_single, plot_match, plot_unmatch


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--dream_idx',default=0, type=int)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Plot the correlation scores of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	

# =============================================================================
# Plot the correlation scores of one dream
# =============================================================================	

# Load the correlation scores
results_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation', 
                           'correlation_scores_'+args.st)
results_list = os.listdir(results_dir)

# Iterations
for dream in tqdm(results_list, desc='dreams'):
    result = np.load(os.path.join(results_dir, dream), allow_pickle=True).item()
    times = result['times']
    correlations = result['correlations']
    match_idx = result['corresponding_img_idx']
    plot_single(args, times, correlations, dream[:-4], match_idx)
    plot_match(args, times, correlations, dream[:-4], match_idx)
    plot_unmatch(args, times, correlations, dream[:-4], match_idx)
    del result, times, correlations, match_idx