def plot_single(args, times, correlations, dream_name, match_idx):
    import os
    from matplotlib import pyplot as plt

    print(dream_name)
    # The total number of images
    num_img = correlations.shape[0]
    
    # Iterate over images
    for img_idx in range(num_img):

        # Generate figure
        fig = plt.figure(figsize=(16, 6))
        gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])
        
        # Temporal correlation results
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot([-11, 1], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
        ax1.plot(times, correlations[img_idx], color='salmon', lw = 0.5)
        ax1.set(xlabel='Time (s)', ylabel='Pearson\'s $r$')
        ax1.set_xlim(left=-11, right=1)
        ax1.set_ylim(bottom=-1, top=1)
        ax1.set_title(f'Temporal correlation scores')

        # Correlation results histogram
        ax2 = plt.subplot(gs[0, 1])
        ax2.hist(correlations[img_idx], bins=20, color='lightskyblue', edgecolor='black')
        ax2.set_title(f'Histogram')
        ax2.set_xlim(-1,1)
        ax2.set(xlabel='Pearson\'s $r$', ylabel='Frequency')
        
        fig.suptitle(f"Correlation scores of {dream_name}-image {img_idx}")

        # Save the figures
        if img_idx in match_idx:
            print('True', img_idx)
            save_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                                    'correlation_plots_'+args.st, 'raw_plots', dream_name, 'match')
        else:
            print('False', img_idx)
            save_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                                    'correlation_plots_'+args.st,'raw_plots', dream_name, 'unmatch')
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        fig_name = dream_name+'-image '+str(img_idx) 
        plt.savefig(os.path.join(save_dir, fig_name))
        plt.close()
        del save_dir, fig

def plot_match(args, times, correlations, dream_name, match_idx):
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    print(dream_name)
    # The total number of images
    num_img = correlations.shape[0]
    
    # Generate figure
    fig = plt.figure(figsize=(16, 6))
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])

    # Set the plot colour spectum
    cmap = "hsv"
    colours = plt.colormaps[cmap](np.linspace(0,1,num_img))
    
    # Iterate over images
    for img_idx in range(num_img):

        if img_idx in match_idx:
            # Temporal correlation results
            ax1 = plt.subplot(gs[0, 0])
            ax1.plot([-11, 1], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
            ax1.plot(times, correlations[img_idx], color=colours[img_idx], lw = 0.5, alpha=0.5)
            ax1.set(xlabel='Time (s)', ylabel='Pearson\'s $r$')
            ax1.set_xlim(left=-11, right=1)
            ax1.set_ylim(bottom=-1, top=1)
            ax1.set_title(f'Temporal correlation scores')

            # Correlation results histogram
            ax2 = plt.subplot(gs[0, 1])
            ax2.hist(correlations[img_idx], bins=20, color=colours[img_idx], edgecolor='black')
            ax2.set_title(f'Histogram')
            ax2.set_xlim(-1,1)
            ax2.set(xlabel='Pearson\'s $r$', ylabel='Frequency')
            
    fig.suptitle(f"Correlation scores of {dream_name} (match)")

    # Save the figures
    save_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                                'correlation_plots_'+args.st, 'raw_plots', dream_name, 'match')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    fig_name = dream_name
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.close()
    del save_dir, fig

def plot_unmatch(args, times, correlations, dream_name, match_idx):
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    print(dream_name)
    # The total number of images
    num_img = correlations.shape[0]
    
    # Generate figure
    fig = plt.figure(figsize=(16, 6))
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])

    # Set the plot colour spectum
    cmap = "hsv"
    colours = plt.colormaps[cmap](np.linspace(0,1,num_img))
    
    # Iterate over images
    for img_idx in range(num_img):

        if img_idx not in match_idx:
            # Temporal correlation results
            ax1 = plt.subplot(gs[0, 0])
            ax1.plot([-11, 1], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
            ax1.plot(times, correlations[img_idx], color=colours[img_idx], lw = 0.2, alpha=0.2)
            ax1.set(xlabel='Time (s)', ylabel='Pearson\'s $r$')
            ax1.set_xlim(left=-11, right=1)
            ax1.set_ylim(bottom=-1, top=1)
            ax1.set_title(f'Temporal correlation scores')

            # Correlation results histogram
            ax2 = plt.subplot(gs[0, 1])
            ax2.hist(correlations[img_idx], bins=20, color=colours[img_idx], edgecolor='black')
            ax2.set_title(f'Histogram')
            ax2.set_xlim(-1,1)
            ax2.set(xlabel='Pearson\'s $r$', ylabel='Frequency')
            
    fig.suptitle(f"Correlation scores of {dream_name} (unmatch)")

    # Save the figures
    save_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                                'correlation_plots_'+args.st, 'raw_plots', dream_name, 'unmatch')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    fig_name = dream_name
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.close()
    del save_dir, fig

def hist(args, scores_dir, dream_name):
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    score = np.load(os.path.join(scores_dir, dream_name), 
                    allow_pickle=True).item()

    # Get the true indice and the mean correlation scores
    true_idx = score['corresponding_img_idx']
    mean_scores = score['mean_correlations']
    del score

    # Separate true and false indice
    mask = np.ones(len(mean_scores), dtype=bool)
    mask[true_idx] = False
    false_scores = mean_scores[mask]
    true_scores = mean_scores[true_idx]
    del mean_scores

    ### Plot the histogram of flase indice ###
    cmap = "hsv"
    colours = plt.colormaps[cmap](np.linspace(0,0.9,len(true_scores)))

    plt.figure()
    plt.hist(false_scores, bins=20, color='grey', label='False images scores')
    for idx, item in enumerate(true_idx):
        plt.plot([true_scores[idx], true_scores[idx]], [0, 500], color=colours[idx], 
            label='true image idx '+str(item))
    plt.ylim(0,300)
    plt.xlabel('mean r scores')
    plt.ylabel('frequency')
    plt.legend(loc='best')

    # Save the figures
    save_dir = os.path.join(args.project_dir, 'results', args.test_dataset, 
                                'correlation_plots_'+args.st, 'histograms')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    fig_name = dream_name[:-4]
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.close()

    ### significance test ###
    significant_values = []
    percentile = np.percentile(false_scores, 95)
    # test results for each image
    for idx, true in zip(true_idx, true_scores):
        if true > percentile:
            significant_values.append([idx, true])
        else:
            pass
    # test result for the mean 
    avg_true = np.mean(true_scores)
    if avg_true > percentile:
        significant_values.append(['mean', avg_true])
    else:
        pass

    # Output
    return significant_values