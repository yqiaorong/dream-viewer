def train_scaler_pca(args):
    """The function train the standardization and pca model on THINGS 
    training images.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    ----------
    scaler : list of models
        the standardization model.
    pca : list of models
        the pca model.    
    all_layers : list
        The list of names of layers.
    layer_names : list
        The list of names of layers.
    """
        
    import os
    import numpy as np
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import KernelPCA

    # Set random seed for reproducible results
    seed = 20200220

    ### Load the feature maps ###
    feats = []
    feats_all = []
    # The dictionaries storing the dnn training feature maps in 3 stages
    fmaps_train = {}
    fmaps_train_1 = {}
    fmaps_train_2 = {}
    # The directory of the dnn training feature maps
    fmaps_dir = os.path.join(args.project_dir, 'eeg_dataset','wake_data',
                             'THINGS_EEG2', 'dnn_feature_maps',
                             'full_feature_maps', args.dnn, 
                            'pretrained-'+str(args.pretrained),
                            'training_images')
    fmaps_list = os.listdir(fmaps_dir)
    fmaps_list.sort()
    for f, fmaps in enumerate(tqdm(fmaps_list, desc='training_images')):
        fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
                             allow_pickle=True).item()
        all_layers = fmaps_data.keys()
        if args.layers == 'all':
            layer_names = ['all_layers']
        elif args.layers == 'single':
            layer_names = all_layers
        for l, dnn_layer in enumerate(all_layers):
            if args.layers == 'all':
                if l == 0:
                    feats = np.reshape(fmaps_data[dnn_layer], -1)
                else:
                    feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], 
                                                        -1))
            elif args.layers == 'single':
                if f == 0:
                    feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
                else:
                    feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
        if args.layers == 'all':
            feats_all.append(feats)
    if args.layers == 'all':
        fmaps_train[layer_names[0]] = np.asarray(feats_all)
        print('The old training fmaps shape',fmaps_train[layer_names[0]].shape)
        fmaps_train_1[layer_names[0]] = []
    elif args.layers == 'single':
        for l, dnn_layer in enumerate(layer_names):
            fmaps_train[dnn_layer] = np.squeeze(np.asarray(feats[l]))
            print('The old training fmaps shape',fmaps_train[dnn_layer].shape)
            fmaps_train_1[dnn_layer] = []

    ### Train the models ###
    # Standardize the data
    scaler = []
    for l, dnn_layer in enumerate(layer_names):
        scaler.append(StandardScaler())
        ## Partial fit
        # The number of subsets
        num_subset = 1654
        # The size of each subset
        chunk_size = 16540/num_subset
        for i in tqdm(range(num_subset),desc='StandardScaler training'):
            scaler[l].partial_fit(fmaps_train[dnn_layer][int(i*chunk_size):
                                                         int(i*chunk_size+
                                                             chunk_size)])
        ## Partial transform
        # The number of full features for each image
        num_img = fmaps_train[dnn_layer].shape[0]
        for i in tqdm(range(num_img), desc='StandardScaler transform'):
            fmaps_train_1[dnn_layer].append(scaler[l].transform(
                fmaps_train[dnn_layer][i].reshape(1,-1)))
        ## Convert lists to array
        # Create transitions directory
        fmaps_train_dir = os.path.join(args.project_dir, 'eeg_dataset','wake_data',
                             'THINGS_EEG2', 'dnn_feature_maps', 'transitions')
        if os.path.isdir(fmaps_train_dir) == False:
            os.makedirs(fmaps_train_dir)
        fmaps_train_2[dnn_layer] = np.memmap(os.path.join(fmaps_train_dir, 
                                                          'transition_data'+dnn_layer), 
                                             dtype='float32', mode='w+', 
                                             shape=fmaps_train[dnn_layer].shape)
        # Write in the data
        for i, features in enumerate(tqdm(fmaps_train_1[dnn_layer], desc='Lists to array')):
            fmaps_train_2[dnn_layer][i,:] = features
        print('The middle training fmaps shape',fmaps_train_2[dnn_layer].shape)
    del fmaps_train, fmaps_train_1

    # Apply PCA
    pca = []
    for l, dnn_layer in enumerate(layer_names):
        print('args.n_components',args.n_components)
        pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
            degree=4, random_state=seed))
        pca[l].fit(fmaps_train_2[dnn_layer])
        fmaps_train_2[dnn_layer] = pca[l].transform(fmaps_train_2[dnn_layer])
        print('The final training fmaps shape',fmaps_train_2[dnn_layer].shape)

    ### Save the downsampled feature maps ###
    save_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data','THINGS_EEG2',
                            'dnn_feature_maps','pca_feature_maps', args.dnn, 
                            'pretrained-'+str(args.pretrained), 'layers-'+args.layers)
    file_name = 'pca_feature_maps_training'
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file_name), fmaps_train_2)
    del fmaps_train_2
    
    return scaler, pca, all_layers, layer_names

def apply_scaler_pca(args, img_category, scaler, pca):
    """The function apply the standardization and pca models on the target 
    images dataset.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.
    img_category : str
        The image category within the used dataset.
    scaler : list of models
        the standardization model.
    pca : list of models
        the pca model.
    """
        
    import os
    import numpy as np
    from tqdm import tqdm

    ### Load the feature maps ###
    feats = []
    feats_all = []
    fmaps_test = {}
    # The full feature maps directory
    if args.dataset == 'THINGS_EEG2':
        fmaps_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',args.dataset,
                                'dnn_feature_maps','full_feature_maps',args.dnn,
                                'pretrained-'+str(args.pretrained),img_category+'_images')
    elif args.dataset == 'SCIP':
        fmaps_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',args.dataset,
                                'dnn_feature_maps','full_feature_maps',args.dnn,
                                'pretrained-'+str(args.pretrained),img_category)
    elif args.dataset == 'Zhang_Wamsley':
        fmaps_dir = os.path.join(args.project_dir,'eeg_dataset','dream_data',args.dataset,
                                'dnn_feature_maps','full_feature_maps',args.dnn,
                                'pretrained-'+str(args.pretrained),img_category)
    elif args.dataset == 'ZW_REMs':
        fmaps_dir = os.path.join(args.project_dir,'eeg_dataset','dream_data','Zhang_Wamsley',
                                'REMs','dnn_feature_maps','full_feature_maps',args.dnn,
                                'pretrained-'+str(args.pretrained),img_category)
    fmaps_list = os.listdir(fmaps_dir)
    fmaps_list.sort()
    for f, fmaps in enumerate(tqdm(fmaps_list, desc=img_category)):
        fmaps_data = np.load(os.path.join(fmaps_dir, fmaps), allow_pickle=True).item()
        all_layers = fmaps_data.keys()
        if args.layers == 'all':
            layer_names = ['all_layers']
        elif args.layers == 'single':
            layer_names = all_layers
        for l, dnn_layer in enumerate(all_layers):
            if args.layers == 'all':
                if l == 0:
                    feats = np.reshape(fmaps_data[dnn_layer], -1)
                else:
                    feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
            elif args.layers == 'single':
                if f == 0:
                    feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
                else:
                    feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
        if args.layers == 'all':
            feats_all.append(feats)
    if args.layers == 'all':
        fmaps_test[layer_names[0]] = np.asarray(feats_all)
    elif args.layers == 'single':
        for l, dnn_layer in enumerate(layer_names):
            fmaps_test[dnn_layer] = np.squeeze(np.asarray(feats[l]))
    
    ### Apply the models on feature maps ###
    # Standardize the data
    for l, dnn_layer in enumerate(layer_names):
        fmaps_test[dnn_layer] = scaler[l].transform(fmaps_test[dnn_layer])
    # Apply PCA
    for l, dnn_layer in enumerate(layer_names):
        fmaps_test[dnn_layer] = pca[l].transform(fmaps_test[dnn_layer])

    ### Save the downsampled feature maps ###
    if args.dataset == 'THINGS_EEG2' or args.dataset == 'SCIP':
        save_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',args.dataset,
                                'dnn_feature_maps','pca_feature_maps', args.dnn, 
                                'pretrained-'+str(args.pretrained), 'layers-'+args.layers)
    elif args.dataset == 'Zhang_Wamsley':
        save_dir = os.path.join(args.project_dir,'eeg_dataset','dream_data',args.dataset,
                                'dnn_feature_maps','pca_feature_maps', args.dnn, 
                                'pretrained-'+str(args.pretrained), 'layers-'+args.layers) 
    elif args.dataset == 'ZW_REMs':
        save_dir = os.path.join(args.project_dir,'eeg_dataset','dream_data','Zhang_Wamsley',
                                'REMs','dnn_feature_maps','pca_feature_maps', args.dnn, 
                                'pretrained-'+str(args.pretrained), 'layers-'+args.layers)       
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    print(save_dir)
    file_name = 'pca_feature_maps_'+img_category
    np.save(os.path.join(save_dir, file_name), fmaps_test)
    del fmaps_test