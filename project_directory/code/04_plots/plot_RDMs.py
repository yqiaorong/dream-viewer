import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--dream_idx',default=0, type=int)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Plot the RDMs of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
# =============================================================================
# Load correlation results
# =============================================================================

# Dream correlation scores list
ZW_corr_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                        'correlation_scores_'+args.st)
dreams_corrs = os.listdir(ZW_corr_dir)

# Load correlation scores
RDMs = []
for c in dreams_corrs:
	result = np.load(os.path.join(ZW_corr_dir, c), allow_pickle=True).item()
	mean_corr = result['mean_correlations']
	RDMs.append(mean_corr)
	del result, mean_corr
RDMs = np.array(RDMs)
print('The shape of RDMs:' , RDMs.shape, ' (number of dreams, number of images)')

# Dream images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:].split('_')[0] for s in dreams_imgs]
	
# =============================================================================
# Plot the full RDMs
# =============================================================================

# Load df
df = pd.read_excel('../project_directory/results/Zhang_Wamsley/df.xlsx')
non_nah_elements = [i for i in df['number_of_images'] if i != 'nah' and i != '-']

fig = plt.figure(figsize=(10, 4))
im = plt.imshow(RDMs, cmap='viridis',
				extent=[0, len(dreams_imgs), 0, len(dreams_corrs)], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

# Horizontal borders
for i in range(len(dreams_corrs)):
	plt.plot([0, len(dreams_imgs)], [i,i], 'k--', lw=0.4)

# Vertical borders
cumulative = 0
dreams = []
for idx, row in df.iterrows():
    if row['dreams_imgs_idx'] != 'nah':
        if row['dreams'] not in dreams:
            dreams.append(row['dreams'])
            num_imgs = int(row['number_of_images'][-1])
            cumulative += num_imgs
            plt.plot([cumulative, cumulative], [0,len(dreams_corrs)], 'k--', lw=0.4)
            del num_imgs

plt.xlim([0,int(len(dreams_corrs[:30])*8)])
plt.ylim([0,int(len(dreams_corrs[:30]))])
plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'unnormalized RDMs')

fig.tight_layout()
plt.show()

# =============================================================================
# Plot the max RDMs
# =============================================================================

fig = plt.figure(figsize=(8, 8))

max_RDMs = np.empty((len(dreams_corrs), len(dreams_corrs)))
for v in range(len(dreams_corrs)):
    previous_cumu = 0
    current_cumu = 0
    for h in range(len(dreams_corrs)):
        num_imgs = int(non_nah_elements[h][-1])
        current_cumu += num_imgs
        # print(f'({v},{h}): {previous_cumu}, {current_cumu}')
        max_RDMs[v,h] = max(RDMs[v, previous_cumu:current_cumu])
        previous_cumu += num_imgs
        del num_imgs
    del previous_cumu, current_cumu

# Normalization
norm_max_RDMs = normalize(max_RDMs)

im = plt.imshow(norm_max_RDMs, cmap='viridis',
				extent=[0, len(dreams_corrs), 0, len(dreams_corrs)], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

plt.xlim([0,len(dreams_corrs)])
plt.ylim([0,len(dreams_corrs)])
plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'max normalized RDMs')

fig.tight_layout()
plt.show()