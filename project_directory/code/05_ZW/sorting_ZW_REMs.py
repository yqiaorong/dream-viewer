import os
import shutil
import argparse
import numpy as np
import pandas as pd

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

# =============================================================================
# Load metadata
# =============================================================================

ZW_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'Zhang_Wamsley')

# Load Records
Records = pd.read_csv(os.path.join(ZW_dir, 'Records.csv'))

raw_CaseID = Records[Records['Last sleep stage'] == 5]['Case ID']
del Records

CaseID = []
for s in raw_CaseID:
    CaseID_part = s.split('_')
    if len(CaseID_part[0]) == 2:
        CaseID_part[0] = '0'+ CaseID_part[0]
    CaseID.append(CaseID_part[0]+CaseID_part[1])
df = pd.DataFrame({'Case ID': CaseID})
del raw_CaseID, CaseID

# Load dream images
ZW_img_dir = os.path.join(ZW_dir, 'images')
img_list = os.listdir(ZW_img_dir)
img_ID = [s[6:].split('_')[0] for s in img_list]
img_uniqID = np.unique(img_ID)

# The Case ID of REM dreams with images with no _
REMID = []
for id in df['Case ID']:
    if id in img_uniqID:
        REMID.append(id)

# The Case ID of REM dreams with images with _
REM_ID = [s[:3] + '_' + s[3:] for s in REMID]

# Get the indices of images of REM dreams in the image list
REM_img_idx = []
for idx, img in enumerate(img_list):
    if img in REMID:
        REM_img_idx.append(idx)

# Create new directory saving all REMs
REM_dir = os.path.join(ZW_dir, 'REMs')
if os.path.isdir(REM_dir) == False:
    os.makedirs(REM_dir)

# =============================================================================
# Tranfer REM images
# =============================================================================

# Create new directory saving all REMs images
REM_img_dir = os.path.join(REM_dir, 'images')
if os.path.isdir(REM_img_dir) == False:
    os.makedirs(REM_img_dir)

# The dictionary storing the REM dreams and corresponding image indices
dict = {}
for id in REMID:
    dict[id] = []
    for i, img in enumerate(img_ID):
        if img == id:
            dict[id].append(i)
    for idx in dict[id]:
        source_dir = os.path.join(ZW_img_dir, img_list[idx])
        desti_dir = os.path.join(REM_img_dir, id)
        if os.path.isdir(desti_dir) == False:
            os.makedirs(desti_dir)
        shutil.copy(source_dir, desti_dir)

# =============================================================================
# Tranfer REM preprocessed EEG data
# =============================================================================

# Create new directory saving all preprocessed REMs EEG data
REM_eeg_dir = os.path.join(REM_dir, 'preprocessed_data')
if os.path.isdir(REM_eeg_dir) == False:
    os.makedirs(REM_eeg_dir)

# Load preprocessed REMs EEG data
prepr_dir = os.path.join(ZW_dir, 'preprocessed_data')
prepr_list = os.listdir(prepr_dir)

for p in prepr_list:
    if p[6:-4] in REM_ID:
        source_dir = os.path.join(prepr_dir, p)
        desti_dir = os.path.join(REM_eeg_dir)
        if os.path.isdir(desti_dir) == False:
            os.makedirs(desti_dir)
        shutil.copy(source_dir, desti_dir)

# =============================================================================
# Extract the REM dream reports
# =============================================================================

# Load Records
Reports = pd.read_csv(os.path.join(ZW_dir, 'Data', 'Reports.csv'))

IDs = []
for id in Reports['Case ID']:
    id_part = id.split('_')
    if len(id_part[0]) == 2:
        id_part[0] = '0'+ id_part[0]
    IDs.append(id_part[0]+'_'+id_part[1])

reports_df = pd.DataFrame({'Case ID': IDs, 'Text of Report': Reports['Text of Report']})
REM_df = reports_df[reports_df['Case ID'].isin(REM_ID)]
del Reports, reports_df

REM_df['CaseID'] = REMID

# Save the REM dreams reports
REM_df.to_excel(os.path.join(ZW_dir, 'REMs', 'REM_reports.xlsx'), index=False)