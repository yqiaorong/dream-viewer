def train_model_THINGS(args):
    """The function trains the encoding model using LogisticRegression. X train 
    is THINGS2 dnn feature maps and Y train is the THINGS2 real EEG training 
    data; and uses the model to predict the test EEG data.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    ----------
    pred_eeg_data_test: array with shape (images/dreams, channels x times)
        The predicted EEG data.
    """

    import os
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    ### Load the training DNN feature maps ###
    # Load the training DNN feature maps directory
    dnn_parent_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                  'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                  args.dnn, 'pretrained-True', 'layers-all')
    # Load the training DNN feature maps (16540, 3000)
    dnn_fmaps_train = np.load(os.path.join(dnn_parent_dir, 'pca_feature_maps_training.npy'), 
                                allow_pickle=True).item()
    
    ### Load the training EEG data ###
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                 'THINGS_EEG2','preprocessed_data')
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,11), desc='THINGS2 subjects'):
        # Load the THINGS2 training EEG data
        data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                            'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        # Get the THINGS2 training channels and times
        if train_subj == 1:
            train_ch_names = data['ch_names']
        else:
            pass
        # Average the training EEG data across repetitions (16540,17,100)
        data = np.mean(data['preprocessed_eeg_data'], 1)
        if args.test_dataset == 'Zhang_Wamsley':
            # Remove the EEG data from 'POz' channel (16540,16,100)
            POz_idx = train_ch_names.index('POz')
            data = np.delete(data,POz_idx,axis=1)
        else: 
            pass
        # Reshape the training EEG data
        # THINGS_EEG1 (16540, 17 x 100) / Zhang_Wamsley (16540, 16 x 100)
        data = np.reshape(data, (data.shape[0],-1))
        eeg_data_train.append(data)
        del data
    # Average the training EEG data across subjects
    # THINGS_EEG1 (16540, 17 x 100) / Zhang_Wamsley (16540, 16 x 100)
    eeg_data_train = np.mean(eeg_data_train,0)

    ### Train the encoding model ###
    # Train the encoding models
    reg = LinearRegression().fit(dnn_fmaps_train['all_layers'],eeg_data_train)

    ### Load the test DNN feature maps ###
    # Load the test DNN feature maps directory
    dnn_parent_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                args.dnn, 'pretrained-True', 'layers-all')
    # Load the test DNN feature maps (images,3000)
    dnn_fmaps_test = np.load(os.path.join(dnn_parent_dir, 'pca_feature_maps_test.npy'
                            ), allow_pickle=True).item()

    ### Predict the EEG test data using the encoding model ###
    # Predict the test EEG data 
    # THINGS_EEG1 (images, 17 x 15) 
    pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])

    ### Output ###
    return pred_eeg_data_test

def test_model_THINGS(args, pred_eeg_data_test, test_subj):
    """The function tests the encoding model by correlating the predicted EEG 
    test data with real EEG test data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    pred_eeg_data_test: array with shape (images/dreams, channels x times)
        The predicted EEG data.
    test_subj : int
        Used test subject.
    
    Returns
    ----------
    encoding_accuracy: list of float
        The encoding accuracy / correlation result.
    test_times : list of float
        EEG time points.
    """

    import os
    import numpy as np
    from scipy.stats import pearsonr as corr

    ### Load the test EEG data ###
    # Load the THINGS1 test EEG data 
    eeg_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data',
                                    'THINGS_EEG1', 'preprocessed_data', 
                                    'sub-'+format(test_subj,'02'))
    eeg_data_test = np.load(os.path.join(eeg_test_dir, 'preprocessed_eeg_test.npy'),
                            allow_pickle=True).item()
    # Get the number of test images
    num_img = eeg_data_test['preprocessed_eeg_data'].shape[0]
    # Get test channel names and times
    test_ch_names = eeg_data_test['ch_names']
    test_times = eeg_data_test['times']
    # Average the test EEG data across repetitions if it's THINGS
    eeg_data_test_avg = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)

    ### Separate the dimension of EEG channels and times ###
    pred_eeg_data_test = np.reshape(pred_eeg_data_test,(num_img,len(test_ch_names),len(test_times)))
    eeg_data_test_avg = np.reshape(eeg_data_test_avg,(num_img,len(test_ch_names),len(test_times)))
    del eeg_data_test
    
    ### Test the encoding model ###
    # Calculate the encoding accuracy
    encoding_accuracy = np.zeros((len(test_ch_names),len(test_times)))
    for t in range(len(test_times)):
        for c in range(len(test_ch_names)):
            encoding_accuracy[c,t] = corr(pred_eeg_data_test[:,c,t],
                eeg_data_test_avg[:,c,t])[0]
    # Average the encoding accuracy across channels
    encoding_accuracy = np.mean(encoding_accuracy,0)
            
    # ### Output ###
    return encoding_accuracy, test_times

def model_ZW(args, dreams_eegs_idx, dreams_imgs_idx, crop_t):
    """The function trains the encoding model using LogisticRegression. X train 
    is the partial Zhang_Wamsley dream image feature maps and Y train is the partial
    Zhang_Wamsley dream real EEG data; and uses the model to predict the test EEG data.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.
    dreams_eegs_idx : list of int
    dreams_imgs_idx : list of lists of int

    Returns
    ----------
    scores : array 
    """

    import os
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr as corr

    ### Load the training DNN feature maps ###
    # Load the training DNN feature maps directory
    dnn_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                                  'Zhang_Wamsley', 'dnn_feature_maps', 'pca_feature_maps', 
                                  args.dnn, 'pretrained-True', 'layers-all')
    # Load the training DNN feature maps (images, 3000)
    dnn_fmaps = np.load(os.path.join(dnn_dir, 'pca_feature_maps_dreams.npy'), 
                                allow_pickle=True).item()
    
    ### Load the Zhang_Wamsley EEG data ###
    # Load the Zhang_Wamsley EEG data directory
    eeg_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                                'Zhang_Wamsley','preprocessed_data')
    # The list of preprocessed EEG
    dreams_eegs = os.listdir(eeg_dir)
    # Iterate over Zhang_Wamsley dreams
    eeg_data = []
    for idx, i in enumerate(dreams_eegs_idx):
        print(i, dreams_eegs[i], dreams_imgs_idx[idx])
        # Load the Zhang_Wamsley EEG data (16, total_samples)
        data = np.load(os.path.join(eeg_dir,dreams_eegs[i]), allow_pickle=True).item()
        data = data['preprocessed_eeg_data']
        # Crop the last 20s (16, samples)
        data = data[:,-crop_t:]
        # duplicate the data according to (number of imgs each dream, 16, samples)
        data = np.tile(data[np.newaxis,:,:], (len(dreams_imgs_idx[idx]), 1, 1))
        eeg_data.append(data)
        del data
    eeg_data = np.vstack(eeg_data) # (number of imgs, 16, samples)
    # Get number of channels and times
    num_ch = eeg_data.shape[1]
    num_t = eeg_data.shape[2]
    # Reshape eeg data # (number of imgs, 16 x samples)
    eeg_data = np.reshape(eeg_data, (eeg_data.shape[0],-1))
 
    ### Train the encoding model ###
    # Select 100 dreams for training 
    select_imgs = dreams_imgs_idx[:100]
    # flat the list of select_imgs
    flat_select_imgs = [item for sublist in select_imgs for item in sublist]
    del select_imgs
    reg = LinearRegression().fit(dnn_fmaps['all_layers'][flat_select_imgs], eeg_data[flat_select_imgs])

    ### Predict the EEG test data using the encoding model ###
    # Predict the test EEG data (not selected imgs, 16 x 2000)
    mask = np.ones(dnn_fmaps['all_layers'].shape[0], dtype=bool)
    mask[flat_select_imgs] = False
    pred_eeg_data_test = reg.predict(dnn_fmaps['all_layers'][mask])

    ### Test the model
    # Reshape the predicted eeg data test (not selected imgs, 16, 2000)
    pred_eeg_data_test = np.reshape(pred_eeg_data_test, (-1, num_ch, num_t))
    # Reshape the eeg data test (not selected imgs, 16, 2000)
    eeg_data_test = np.reshape(eeg_data[mask], (-1, num_ch, num_t)) 
    # Compute correlation scores
    scores = np.empty((num_ch, num_t))
    for c in range(num_ch):
        for t in range(num_t):
            scores[c,t] = corr(pred_eeg_data_test[:,c,t], eeg_data_test[:,c,t])[0]

    ### Output ###
    return scores