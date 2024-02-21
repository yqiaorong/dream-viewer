def corr_s(args, pred_eeg_data_test, eeg_idx, img_idx, crop_t, REM=False):
    """The function tests the encoding model by correlating the predicted EEG 
    test data with real EEG test data. This method computes the correlation
    scores by applying the predicted data on each time point in the dream.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    pred_eeg_data_test: array with shape (images/dreams, channels x times)
        The predicted EEG data.
    eeg_idx : int
        The dream index in dreams preprocessed eeg list.
    img_idx : int
        The image index in dream images list.
    crop_t : int
        The number of samples before waking, samples = time x sample rate 100.
    
    Returns
    ----------
    corr_score : array
        The array storing the temporal correlation scores of all images to one 
        dream.
    mean_score : array
        The array storing the mean correlation scores of all images to one dream.
    """

    import os
    import numpy as np
    from scipy.stats import pearsonr as corr
    
    ### Load the test EEG data ###
    # Load the Zhang_Wamsley test EEG directory
    if REM == False:
        eeg_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data',
                                    args.test_dataset, 'preprocessed_data')
    else:
        eeg_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data',
                                    args.test_dataset, 'REMs','preprocessed_data')
    eeg_test_list = os.listdir(eeg_test_dir)
    # Load the Zhang_Wamsley test EEG data (16, total time)
    eeg_data_test = np.load(os.path.join(eeg_test_dir, eeg_test_list[eeg_idx]),
                            allow_pickle=True).item()
    eeg_data_test = eeg_data_test['preprocessed_eeg_data']
    # Crop the test EEG data of last crop_t/100 s (16, crop_t)
    eeg_data_test = eeg_data_test[:,-crop_t:]

    ### Test the encoding model ###
    # Get the number of time points
    num_t = eeg_data_test.shape[1]
    # The array of correlation score of one image
    corr_score = np.empty((num_t))
    # Compute the correlation scores
    for t in range(num_t):
        corr_score[t] = corr(pred_eeg_data_test[img_idx], eeg_data_test[:,t])[0]
    # Compute the mean correlation score
    mean_score = np.mean(corr_score)

    ### Output ###
    return corr_score, mean_score

def corr_t(args, pred_eeg_data_test, eeg_idx, img_idx, crop_t):
    """The function tests the encoding model by correlating the predicted EEG 
    test data with real EEG test data. This method computes the correlation
    scores by applying the predicted data as a sliding window in the dream.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    pred_eeg_data_test: array with shape (images/dreams, channels x times)
        The predicted EEG data.
    eeg_idx : int
        The dream index in dreams list.
    img_idx : int
        The image index in dream images list.
    crop_t : int
        The number of samples before waking, samples = time x sample rate 100.
    
    Returns
    ----------
    corr_score : array
        The array storing the temporal correlation scores of all images to one 
        dream.
    mean_score : array
        The array storing the mean correlation scores of all images to one dream.
    """
    
    import os
    import numpy as np
    from scipy.stats import pearsonr as corr
    
    ### Load the test EEG data ###
    # Load the Zhang_Wamsley test EEG directory
    eeg_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data',
                                    args.test_dataset, 'preprocessed_data')
    eeg_test_list = os.listdir(eeg_test_dir)
    # Load the Zhang_Wamsley test EEG data (16, total time)
    eeg_data_test = np.load(os.path.join(eeg_test_dir, eeg_test_list[eeg_idx]),
                            allow_pickle=True).item()
    eeg_data_test = eeg_data_test['preprocessed_eeg_data']
    # Crop the test EEG data of last crop_t/100 s (16, crop_t)
    eeg_data_test = eeg_data_test[:,-crop_t:]

    ### Test the encoding model ###
    # Get the number of time points
    num_t = eeg_data_test.shape[1]
    # Get the number of effective time points
    eff_t = num_t - 15
    # The array of correlation score of one image
    corr_score = np.empty((eff_t))
    # Compute the correlation scores
    for t in range(eff_t):
        corr_score[t] = corr(pred_eeg_data_test[img_idx], eeg_data_test[:,t:t+15].ravel())[0]
    # Compute the mean correlation score
    mean_score = np.mean(corr_score)

    ### Output ###
    return corr_score, mean_score

def feature_selection(args, feature_maps_train, eeg_data_train):
    """The function reduces the dimension of features, trains and tests the 
    encoding model. 

    Parameters
    ----------
    args : Namespace
        Input arguments.
    feature_maps_train: array with shape (samples, features)
        The dnn training feature maps. 
    eeg_data_train : array with shape (samples, 1)
        The training EEG data.

    Returns
    ----------
    pred_eeg : array with shape (samples,)
        The predicted EEG data. 
    """
    
    import os
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import SelectKBest, f_regression

    ### Feature selection ###
    num_feat = 300
    # Feature selection model
    best_feat = SelectKBest(f_regression, k=num_feat).fit(feature_maps_train, 
                                                          eeg_data_train)
    # Transfer the training features
    train = best_feat.transform(feature_maps_train)
    del feature_maps_train
    ### Train the encoding model ###
    reg = LinearRegression().fit(train, eeg_data_train)
    
    ### Load the test dream DNN feature maps ###
    ZW_rem_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                                args.test_dataset, 'REMs')
    # Load the test DNN feature maps directory
    dnn_test_dir = os.path.join(ZW_rem_dir, 'dnn_feature_maps', 
                                'pca_feature_maps', args.dnn, 'pretrained-True', 
                                'layers-all')
    dnn_test_list = os.listdir(dnn_test_dir)
    # Load the test DNN feature maps 
    pred_eeg = []
    for fmap in dnn_test_list:
        dnn_fmaps_test = np.load(os.path.join(dnn_test_dir,fmap,
                                ), allow_pickle=True).item()
        # Feature selection
        dnn_fmaps_test['all_layers'] = best_feat.transform(dnn_fmaps_test['all_layers'])
        # Predict the EEG test data using the encoding model 
        test = reg.predict(dnn_fmaps_test['all_layers'])
        pred_eeg.append(test)
        del dnn_fmaps_test['all_layers'], test
    pred_eeg = np.array(pred_eeg).ravel()
    
    return pred_eeg