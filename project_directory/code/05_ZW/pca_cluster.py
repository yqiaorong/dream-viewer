"""apply the PCA on Zhang & Wamsley REM EEG data and conduct 
the clustering of dreams.

Parameters
----------
project_dir : str
	Directory of the project folder.
"""

import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
args = parser.parse_args()

print(f'>>> Apply PCA and clustering on Zhang & Wamsley REM EEG <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load preprocessed EEG data and apply StandardScaler
# =============================================================================

# Load the directory
ZW_eeg_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                        'Zhang_Wamsley', 'REMs', 'preprocessed_data')
ZW_eeg_list = os.listdir(ZW_eeg_dir)

# Standardize the data
scaler = StandardScaler()
for i in ZW_eeg_list:
    # Load preprocessed eeg data
    eeg = np.load(os.path.join(ZW_eeg_dir, i), allow_pickle=True).item()
    eeg = eeg['preprocessed_eeg_data']
    # Crop the last 10s
    eeg = eeg[:,-1000:]
    # Partial fit
    scaler.partial_fit(eeg)
    del eeg
# Transform
standard_eegs = []
for i in ZW_eeg_list:
    # Load preprocessed eeg data
    eeg = np.load(os.path.join(ZW_eeg_dir,i), allow_pickle=True).item()
    eeg = eeg['preprocessed_eeg_data']
    # Crop the last 10s
    eeg = eeg[:,-1000:]
    # Transform
    standard_eegs.append(scaler.transform(eeg))
    del eeg
standard_eegs = np.array(standard_eegs)
# Average across time
standard_eegs = np.mean(standard_eegs, axis=2)
print('standardscaler:', standard_eegs.shape)


# =============================================================================
# Apply PCA
# =============================================================================

# Set random seed for reproducible results
seed = 20200220
pca = KernelPCA(n_components=2, kernel='poly', degree=4, random_state=seed)
pca.fit(standard_eegs)
pca_eegs = pca.transform(standard_eegs)
print('pca:', pca_eegs.shape)
del standard_eegs


# =============================================================================
# Clustering
# =============================================================================


kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto', random_state=seed)
kmeans.fit(pca_eegs)
print(kmeans.labels_)


# =============================================================================
# Plot
# =============================================================================

colours = ['firebrick', 'salmon', 'orange', 'gold', 'palegreen', 'yellowgreen',
           'forestgreen', 'lightskyblue', 'cornflowerblue', 'mediumpurple', 'hotpink' ]

fig = plt.figure(figsize=(6, 6))
plt.title('REM dreams clusters')
for i, name in enumerate(ZW_eeg_list):
    plt.scatter(pca_eegs[i,0], pca_eegs[i,1], color=colours[i], label=name[6:-4]+'_'+str(i))
plt.legend()
plt.show()