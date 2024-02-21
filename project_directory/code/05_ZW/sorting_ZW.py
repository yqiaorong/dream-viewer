"""Create the dataframe sorting labels in Zhang & Wamsley dataset"""
import os
import argparse
import pandas as pd
from collections import OrderedDict

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# =============================================================================
# Load directories
# =============================================================================

# Dreams images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:] for s in dreams_imgs]
dreams_imgs_short = [s.split('_')[0] for s in dreams_imgs]

# Dreams EEGs list
ZW_EEG_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'preprocessed_data')
dreams_eegs = os.listdir(ZW_EEG_dir)
dreams_eegs = [s[6:-4].replace('_', '') for s in dreams_eegs]

# Load Records
ZW_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley')
report = pd.read_csv(os.path.join(ZW_dir, 'Records.csv')) 

CaseID = []
for s in report['Case ID']:
    CaseID_part = s.split('_')
    if len(CaseID_part[0]) == 2:
        CaseID_part[0] = '0'+ CaseID_part[0]
    CaseID.append(CaseID_part[0]+CaseID_part[1])

sleep_stage = report['Last sleep stage']

# =============================================================================
# Create the DataFrame
# =============================================================================

eegs = []
eegs_idx = []
imgs = []
imgs_idx = []

# Add dream indice, dream names, dream image indices and dream image names to
# the DataFrame
for eeg_idx, eeg in enumerate(dreams_eegs):
    if eeg in dreams_imgs_short:
        for img_idx, img in enumerate(dreams_imgs):
            if img.split('_')[0] == eeg:
                eegs.append(eeg)
                eegs_idx.append(eeg_idx)
                imgs.append(img)
                imgs_idx.append(img_idx)
    else:
        eegs.append(eeg)
        eegs_idx.append(eeg_idx)
        imgs.append('nah')
        imgs_idx.append('nah')

df = pd.DataFrame({'dreams_idx':eegs_idx, 'dreams':eegs, 
                   'dreams_imgs_idx': imgs_idx, 'dreams_imgs':imgs})

# Add sleep stages to the DataFrame
stages = []
for dream in df['dreams']:
    stages.append(sleep_stage[CaseID.index(dream)])
df['sleep_stages'] = stages

# Add number of images for each dream
occurrences = OrderedDict()
for item in df['dreams']:
    if item in occurrences:
        occurrences[item] += 1
    else:
        occurrences[item] = 1
        
num_occur = []
for idx, row in df.iterrows():
    if row['dreams_imgs_idx'] != 'nah':
        item = row['dreams']
        if item+'_'+str(occurrences[item]) not in num_occur:
            num_occur.append(item+'_'+str(occurrences[item]))
        else:
            num_occur.append('-')
    else:
        num_occur.append('nah')
df['number_of_images'] = num_occur

# Save the dataframe
df.to_excel(os.path.join(args.project_dir, 'results', 'Zhang_Wamsley', 
                         'df.xlsx'), index=False)