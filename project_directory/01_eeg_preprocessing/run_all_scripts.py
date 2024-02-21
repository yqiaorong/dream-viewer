import os

# =============================================================================
# THINGS1 EEG preprocessing
# =============================================================================
for s in range(1,51):
    os.system(f'python ../project_directory/github_code/01_eeg_preprocessing/THINGS1.py --subj {s}')

# =============================================================================
# THINGS2 EEG preprocessing
# =============================================================================
for s in range(1,11):
    os.system(f'python ../project_directory/github_code/01_eeg_preprocessing/THINGS2.py --subj {s}')

# =============================================================================
# Zhang & Wamsley dream EEG preprocessing
# =============================================================================
# Load the eeg PSG dir
PSGs_dir = '../project_directory/eeg_dataset/dream_data/Zhang_Wamsley/Data/PSG/'
PSGs = os.listdir(PSGs_dir)
# Iterate through all PSGs
for PSG in PSGs:
    os.system(f'python ../project_directory/github_code/01_eeg_preprocessing/ZW.py --PSG {PSG[7:-4]}')