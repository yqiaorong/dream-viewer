def epoching(args):
    """The function preprocesses the raw EEG file: channel selection, 
    creating annotations and events, re-reference, bandpass filter,
    epoching, baseline correction and frequency downsampling. Then, it 
    sorts the test EEG data according to the image conditions.

    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    -------
    sort_data : array of shape (image,repetition,channel,time)
        Epoched EEG test data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.
    """

    import os
    import mne
    import numpy as np
    import pandas as pd
    
    ### Load the THINGS1 subject metadata ### 
    # Load the THINGS1 subject directory
    TH1_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',
                           'THINGS_EEG1','raw_data','sub-'+format(args.subj,'02'),
                           'eeg')
    # Load the THINGS1 subject metadata
    dftsv = pd.read_csv(os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                                     '_task-rsvp_events.tsv'), delimiter='\t')
    
    ### Crop the THINGS1 subject metadata ###
    # Select the main 22248 images
    dftsv = dftsv.iloc[:22248]
    # Select events relevant information
    dftsv = dftsv[['onset','object']] 
    
    ### Load the THINGS1 subject EEG data ###
    # Load the THINGS1 subject EEG directory
    TH1_EEG_dir = os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                               '_task-rsvp_eeg.vhdr')
    # Load the THINGS1 subject EEG raw file
    raw = mne.io.read_raw_brainvision(TH1_EEG_dir, preload=True)
    
    ### channel selection ###
    # Pick the main 64 channels
    if args.subj in [49, 50]:
        raw = raw.pick(raw.info['ch_names'][:63])
    else:
        pass
    # Pick up occipital and parietal channels
    chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
                                                   '^O *|^P *'))
    new_chans = [raw.info['ch_names'][c] for c in chan_idx]
    # Pick occipital channels
    raw.pick(new_chans)
    
    ### Create annotations and events ###
    # Annotation onset
    onset = dftsv['onset'] # in seconds
    # Annotation duration
    duration = [0.05]*len(dftsv) # in seconds, too
    # Create annotations
    annot = mne.Annotations(onset=onset, duration=duration, 
                            description=['images']*len(dftsv))
    # Set annotations
    raw.set_annotations(annot)
    # Create events
    events, _ = mne.events_from_annotations(raw)
    
    ### Re-reference and bandpass filter all channels ###
    # Re-reference raw 'average'
    raw.set_eeg_reference()  
    # Bandpass filter
    raw.filter(l_freq=0.1, h_freq=100)
    
    ### Epoching, baseline correction and resampling ###
    # Epoching
    epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0), 
                        preload=True)
    del raw
    # Resampling
    epochs.resample(args.sfreq)
    
    ### Get epoched channels and times ###
    ch_names = epochs.info['ch_names']
    times = epochs.times
    
    ### Sort epoched data according to the THINGS2 test images ###
    # Get epoched data
    epoched_data = epochs.get_data()
    del epochs
    # THINGS2 test images directory
    test_img_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',
                                 'THINGS_EEG2','image_set','test_images')
    # Create list of THINGS2 test images
    test_imgs = os.listdir(test_img_dir)
    # The sorted epoched data
    sort_data = []
    # Iterate over THINGS2 test images
    for test_img in test_imgs:
        # Get the indices of test image 
        indices = dftsv.index[dftsv['object'] == test_img[6:]]
        # Get the data of test image 
        data = [epoched_data[i, :, :] for i in indices]
        # Convert list to array
        data = np.array(data)
        # Add the data to the test THINGS1 EEG data
        sort_data.append(data)
        del indices, data
    # Convert list to array
    sort_data = np.array(sort_data)

    ### Outputs ###
    return sort_data, ch_names, times

def mvnn(args, epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch of each image condition), and then average them 
    across image conditions. The inverse of the resulting averaged covariance
    matrix is used to whiten the EEG data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    epoched_data : array of shape (image,repetition,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    whitened_data = []
    # Notations
    img_cond = epoched_data.shape[0]
    num_rep = epoched_data.shape[1]
    num_ch = epoched_data.shape[2]
    num_time = epoched_data.shape[3]

    ### Compute the covariance matrices ###
    # Covariance matrix of shape:
    # EEG channels × EEG channels
    sigma = np.empty((num_ch, num_ch))
    # Image conditions covariance matrix of shape:
    # Image conditions × EEG channels × EEG channels
    sigma_cond = np.empty((img_cond, num_ch, num_ch))
    # Iterate across the time points
    for i in tqdm(range(img_cond)):
        cond_data = epoched_data[i]
        # Compute covariace matrices at each time point, and then
        # average across time points
        if args.mvnn_dim == "time":
            sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                shrinkage='auto') for t in range(num_time)],
                axis=0)
        # Compute covariace matrices at each epoch, and then 
        # average across epochs
        elif args.mvnn_dim == "epochs":
            sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                shrinkage='auto') for e in range(num_rep)],
                axis=0)
    # Average the covariance matrices across image conditions
    sigma = sigma_cond.mean(axis=0)
    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

    ### Whiten the data ###
    whitened_data = np.reshape((np.reshape(epoched_data, 
    (-1,num_ch,num_time)).swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2), 
    epoched_data.shape)

    ### Output ###  
    return whitened_data

def save_prepr(args, whitened_data, ch_names, times, seed):
    """Shuffle the EEG repetitions across sessions and reshaping the 
    data to the format:
    Image conditions × EGG repetitions × EEG channels × EEG time points.
    Then, the data of both test EEG partitions is saved.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.
    seed : int
        Random seed.

    """
    
    import os
    import numpy as np
    from sklearn.utils import shuffle
    
    ### Save the data ###
    # Notation
    num_rep = whitened_data.shape[1]
    # Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, num_rep), random_state=seed)
    whitened_data = whitened_data[:,idx]
    # Insert the data into a dictionary
    data_dict = {
        'preprocessed_eeg_data': whitened_data,
        'ch_names': ch_names,
        'times': times
    }
    del whitened_data
    # Saving directories
    save_dir = os.path.join(args.project_dir, 'eeg_dataset','wake_data','THINGS_EEG1',
        'preprocessed_data', 'sub-'+format(args.subj,'02'))
    file_name_test = 'preprocessed_eeg_test.npy'
    # Create the directory if not existing and save the data
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name_test), data_dict)
    del data_dict