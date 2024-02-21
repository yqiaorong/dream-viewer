import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory',type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--dream_idx',default=0, type=int)
parser.add_argument('--feature_selection',default=False,type=bool)
args = parser.parse_args()

print(f'>>> Plot the RDMs of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
# =============================================================================
# Load correlation results
# =============================================================================

ZW_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'Zhang_Wamsley')

# Dream correlation scores list
if args.feature_selection == True:
	ZW_corr_dir = os.path.join(ZW_dir, 'REMs', 'results', 'correlation_scores_sf')
else:
    ZW_corr_dir = os.path.join(ZW_dir, 'REMs', 'results', 'correlation_scores_s')
dreams_corrs = os.listdir(ZW_corr_dir)

# Load correlation scores
img_indice = []
RDMs = []
for c in dreams_corrs:
	result = np.load(os.path.join(ZW_corr_dir, c), allow_pickle=True).item()
	# Append indice
	img_idx = result['corresponding_img_idx']
	img_indice.append(img_idx)
	# Append correlation scores
	mean_corr = result['mean_correlations']
	RDMs.append(mean_corr)
	del result, img_idx, mean_corr
RDMs = np.array(RDMs)
print(RDMs.shape)

# Dream images list
ZW_img_dir = os.path.join(ZW_dir, 'REMs','images')
dreams = os.listdir(ZW_img_dir)

	
# =============================================================================
# Plot the full RDMs
# =============================================================================

fig = plt.figure(figsize=(10, 3))
im = plt.imshow(RDMs, cmap='viridis',
				extent=[0, RDMs.shape[1], 0, RDMs.shape[0]], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

# Horizontal borders
for i in range(RDMs.shape[0]):
	plt.plot([0, RDMs.shape[1]], [i,i], 'k--', lw=0.4)

print(img_indice)
# Vertical borders
for img in img_indice:
	plt.plot([int(img[-1]+1),int(img[-1]+1)], [0, RDMs.shape[0]], 'k--', lw=0.4)

plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'REMs RDMs')
fig.tight_layout()
plt.show()

# =============================================================================
# Plot the max RDMs
# =============================================================================

# Get the maximum value among 8 images
max_RDMs = np.empty((RDMs.shape[0], RDMs.shape[0]))
for v in range(RDMs.shape[0]):
	for h in range(RDMs.shape[0]):
		# print(len(RDMs[v, img_indice[h][0]:int(img_indice[h][-1]+1)]))
		# print(f'({v},{h}): {img_indice[h][0]}, {int(img_indice[h][-1]+1)}')
		max_RDMs[v,h] = max(RDMs[v, img_indice[h][0]:int(img_indice[h][-1]+1)])
		if h == v:
			print(RDMs[v, img_indice[h][0]:int(img_indice[h][-1]+1)])

# Normalization
norm_max_RDMs = normalize(max_RDMs)

# Plot original maximum RDMs
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(max_RDMs, cmap='viridis',
				extent=[0, RDMs.shape[0], 0, RDMs.shape[0]], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

plt.xlim([0,RDMs.shape[0]])
plt.ylim([0,RDMs.shape[0]])
plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'max REMs RDMs')

fig.tight_layout()
plt.show()

# Plot normalized maximum RDMs
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(norm_max_RDMs, cmap='viridis',
				extent=[0, RDMs.shape[0], 0, RDMs.shape[0]], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

plt.xlim([0,RDMs.shape[0]])
plt.ylim([0,RDMs.shape[0]])
plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'normalized max REMs RDMs')

fig.tight_layout()
plt.show()