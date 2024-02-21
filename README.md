# dream-viewer

Attempted using electroencephalogram (EEG) to decode visual dream content by leveraging an encoding model which generates EEG signals from deep neural network (DNN) feature maps of visual images. The encoding model is trained on THINGS2 EEG dataset and tested on both THINGS1 waking EEG dataset and Zhang_Wamsley dream EEG dataset. The project pipeline integrates the code from THINGS2 and derives the remaining analyses of this project from it.

## Datasets

The open datasets currently used are：

* [THINGS EEG1](https://www.nature.com/articles/s41597-021-01102-7) 

* [THINGS EEG2 dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

* [Zhang & Wamsley dataset](https://onlinelibrary.wiley.com/doi/10.1111/psyp.13368)

The path of download:

1. [THINGS EEG1](https://openneuro.org/datasets/ds003825/versions/1.2.0): ../project_directory/eeg_dataset/wake_data/THINGS_EEG1/

2. THINGS EEG2: 
   
   * [Raw EEG data](https://osf.io/crxs4/): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/raw_data/

   * [Image set](https://osf.io/y63gw/): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/image_set/

2. THINGS images meta data: 

   * [category](https://osf.io/vehr3): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/

   * [object properties ratings](https://osf.io/3kwn2): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/

3. [Zhang & Wamsley 2019](https://doi.org/10.6084/m9.figshare.22226692): ../project_directory/eeg_dataset/dream_data/Zhang_Wamsley

## The analysis

The files ended with '_func.py' have no needs to be run.

### 01_eeg_preprocessing

* run_all_scripts.py (This preprocesses all datasets, including THINGS1, THINGS2, Zhang & Wamsley.)

* THINGS1.py (This file preprocesses one specific subject of THINGS1.)

* THINGS2.py (This file preprocesses one specific subject of THINGS2.)

* ZW.py (This file preprocesses one specific subject of Zhang & Wamsley.)

### 02_dnn_fmaps

Step 1: run either of followings:

* alexnet_THINGS.py (This file extracts dnn alexnet feature maps from THINGS images.)

* alexnet_ZW.py (This file extracts dnn alexnet feature maps from Zhang & Wamsley images.)

* sort_ZW_REMs.py + alexnet_ZW_REMs.py (This file extracts dnn alexnet feature maps from Zhang & Wamsley REM images.)

Step 2: run the following: 

* pca.py (This file applies StandardScaler and PCA on the full feature maps.)

### 03_validation_test

* encoding_THINGS1.py (This file trains the EEG encoding model on THINGS EEG2 dataset and tests the EEG encoding model on THINGS1 dataset. The correlation result is plotted.)

* ZW.py (This file trains the EEG encoding model and tests the EEG encoding model on Zhang & Wamsley dream dataset. The correlation result is plotted.)

* encoding_ZW.py (This file trains the EEG encoding model on THINGS EEG2 dataset and tests the EEG encoding model on Zhang & Wamsley dataset using the 'spatial correlation method'. The correlation scores of each dream is saved.)

* encoding_ZW_REM.py (This file trains the EEG encoding model on THINGS EEG2 dataset and tests the EEG encoding model on REM dreams Zhang & Wamsley dataset using the 'spatial correlation method'. The correlation scores of each dream is saved.)

* encoding_ZW_REM_fs.py (This file trains the EEG encoding model on THINGS EEG2 dataset and tests the EEG encoding model on REM dreams Zhang & Wamsley dataset using the 'spatial correlation method' with extra feature selections.)

### 04_plots

* plot.py (This file plots the correlation scores for individual dream.)

* plot_hist.py (This file plots the histogram of correlation scores for all dream.)

* plot_RDMs.py (This file plots the RDMs of dreams with feature maps.)

* plot_RDMs_REM.py (This file plots the RDMs of REM dreams with feature maps.)

* plot_stages.py (This file plots the statistical decoding results from all dreams with different sleep stages.)

* plot_meanall.py (This file plots the statistical decoding results from all dreams.)

### 05_ZW

No needs to run the following scripts.

* sorting_ZW.py (THis file sorts all metadata of dreams with feature maps.)

* sorting_ZW_REMs.py (THis file sorts all metadata of REM dreams with feature maps.)