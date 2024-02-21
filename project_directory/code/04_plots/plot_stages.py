import os
import argparse
import numpy as np
import pandas as pd


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--percentile',default=0.95, type=float)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Group the decoding results of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
      

# =============================================================================
# Load the outputs
# =============================================================================

# Load the directory
scores_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
						'correlation_plots_'+args.st)

scores = np.load(os.path.join(scores_dir, '0.95_significant_r_scores.npy'),
                allow_pickle=True).item()


# =============================================================================
# Group the outputs according to the sleep stages
# =============================================================================

# Load the metadata
ZW_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                      args.test_dataset)

records = pd.read_csv(os.path.join(ZW_dir, 'Records.csv'))

# Extract the CasID and the sleep stage
CaseID = []
for s in records['Case ID']:
    CaseID_part = s.split('_')
    if len(CaseID_part[0]) == 2:
        CaseID_part[0] = '0'+ CaseID_part[0]
    CaseID.append(CaseID_part[0]+CaseID_part[1])

sleep_stage = records['Last sleep stage']

# Set the variables for single decoding
N0_yes, N1_yes, N2_yes, N3_yes, N5_yes = 0, 0, 0, 0, 0 
N0_no, N1_no, N2_no, N3_no, N5_no = 0, 0, 0, 0, 0 
for case, stage in zip(CaseID, sleep_stage):
    if case in scores.keys():
        if len(scores[case]) != 0:
            if stage == 0:
                N0_yes += 1
            elif stage == 1:
                N1_yes += 1
            elif stage == 2:
                N2_yes += 1
            elif stage == 3:
                N3_yes += 1
            elif stage == 5:
                N5_yes += 1
        else:
            if stage == 0:
                N0_no += 1
            elif stage == 1:
                N1_no += 1
            elif stage == 2:
                N2_no += 1
            elif stage == 3:
                N3_no += 1
            elif stage == 5:
                N5_no += 1

# Set the variables for mean decoding
N0_myes, N1_myes, N2_myes, N3_myes, N5_myes = 0, 0, 0, 0, 0 
N0_mno, N1_mno, N2_mno, N3_mno, N5_mno = 0, 0, 0, 0, 0 
for case, stage in zip(CaseID, sleep_stage):
    if case in scores.keys():
        if len(scores[case]) != 0 and scores[case][-1][0] == 'mean':
            if stage == 0:
                N0_myes += 1
            elif stage == 1:
                N1_myes += 1
            elif stage == 2:
                N2_myes += 1
            elif stage == 3:
                N3_myes += 1
            elif stage == 5:
                N5_myes += 1
        else:
            if stage == 0:
                N0_mno += 1
            elif stage == 1:
                N1_mno += 1
            elif stage == 2:
                N2_mno += 1
            elif stage == 3:
                N3_mno += 1
            elif stage == 5:
                N5_mno += 1

# Save the group results in dataframe
df = pd.DataFrame({'Last sleep stage':[0,1,2,3,5], 
                   'Decoding':[N0_yes, N1_yes, N2_yes, N3_yes, N5_yes], 
                   'No decoding': [N0_no, N1_no, N2_no, N3_no, N5_no],
                   'Mean decoding': [N0_myes, N1_myes, N2_myes, N3_myes, N5_myes],
                   'No mean decoding': [N0_mno, N1_mno, N2_mno, N3_mno, N5_mno]})
print(df)
# Save the dataframe
df.to_excel(os.path.join(args.project_dir, 'results', 'Zhang_Wamsley', 
                         'correlation_plots_'+args.st, str(args.percentile)+'_decoding_stats.xlsx'), 
                         index=False)