import os
import argparse
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='project_directory', type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--percentile',default=0.95, type=float)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Plot the mean distribution of r scores of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	

# =============================================================================
# Load the data
# =============================================================================

# Load the directory
scores_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation',
						'correlation_scores_'+args.st)
scores_list = os.listdir(scores_dir)
mean_all = []
all = []
for dream_name in scores_list:
    score = np.load(os.path.join(scores_dir, dream_name), 
                    allow_pickle=True).item()
    
    # Get the true indice and the mean correlation scores
    true_idx = score['corresponding_img_idx']
    mean_scores = score['mean_correlations']
    true_scores = mean_scores[true_idx]
    del score

    # Get the mean score
    avg_true = np.mean(true_scores)
    mean_all.append(avg_true)
    all.append(true_scores)
all = list(chain(*all))


# =============================================================================
# Plot the mean histograms
# =============================================================================

# Calculate the percentile
mpval_95 = np.percentile(mean_all, 95)
mpval_5 = np.percentile(mean_all, 5)
print(f'The statistical mean correlation score: {np.mean(mean_all)} with C. I. {mpval_5}/{mpval_95}')
# Plot the mean
plt.figure()
plt.hist(mean_all, bins=20, color='lightskyblue', label='Mean true r scores')
plt.plot([mpval_5, mpval_5], [0, 30], color='salmon', label = '5% percentile')
plt.plot([mpval_95, mpval_95], [0, 30], color='palegreen', label = '95% percentile')
plt.xlabel('r scores')
plt.ylabel('frequency')
plt.legend(loc='best')

# Save the figures
save_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation',
                            'correlation_plots_'+args.st)
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
fig_name = 'mean_true_r_scores'
plt.savefig(os.path.join(save_dir, fig_name))
plt.close()

# Calculate the percentage above zero
mean_per = len([i for i in mean_all if i > 0])/len(mean_all)
print('The percentage of mean true r scores above 0: ', mean_per)

# =============================================================================
# Plot the mean histograms
# =============================================================================

# Calculate the percentile
pval_95 = np.percentile(all, 95)
pval_5 = np.percentile(all, 5)
print(f'The statistical correlation score: {np.mean(all)} with C. I. {pval_5}/{pval_95}')
# Plot the mean
plt.figure()
plt.hist(all, bins=20, color='lightskyblue', label='true r scores')
plt.plot([pval_5, pval_5], [0, 200], color='salmon', label = '5% percentile')
plt.plot([pval_95, pval_95], [0, 200], color='palegreen', label = '95% percentile')
plt.xlabel('r scores')
plt.ylabel('frequency')
plt.legend(loc='best')

# Save the figures
save_dir = os.path.join(args.project_dir, 'results', f'{args.test_dataset}_correlation',
                            'correlation_plots_'+args.st)
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
fig_name = 'true_r_scores'
plt.savefig(os.path.join(save_dir, fig_name))
plt.close()

# Calculate the percentage above zero
per = len([i for i in all if i > 0])/len(all)
print('The percentage of true r scores above 0: ', per)