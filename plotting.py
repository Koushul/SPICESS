import cuml
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)

def plot_latent(
    data,
    labels,
    names=['Gene\nEmbeddings', 'Protein\nEmbeddings'],
    legend=False,
    remove_outliers=False,
    n_components=2,
    separate_dim=False,
    square=False,
    method='umap',
    n_neighbors=None,
    seed=42,
    reduce_only=False,
    save=None,
    colors=None,
    size=100,
    fmt='svg',
):
    method_names = {'pca': 'PC', 'umap': 'UMAP'}
    axs = []
    datax = []
    
    if colors is None:
        colors = ["#4a3e38",
                "#9fbfa2",
                "#b99bbd",
                "#53366c",
                "#b2834b",
                "#00612e",
                "#00acef",
                "#00eee9",
                "#ba4850",
                "#cac84d",
                "#ce52ad",
                "#76d05c",
                "#794bc8",
                "#e85a00"]
    
    plt.rcParams['figure.figsize'] = (16, 8)

    for i, (dat, lab) in enumerate(zip(data, labels)):
        ax = plt.gcf().add_subplot(1, len(data), i+1, projection=None)
        axs.append(ax)
        if i == 0 or separate_dim:
            red = cuml.UMAP(
                n_components=n_components,
                n_neighbors=min(200, dat.shape[0] - 1) if n_neighbors is None else n_neighbors,
                min_dist=.5,
                random_state=42)
            if separate_dim:
                red.fit(dat)
            else:
                red.fit(np.concatenate(data, axis=0))
        plot_data = red.transform(dat)
        datax.append(plot_data)
        
        unique_labels = np.unique(np.concatenate(labels))
        for ix, l in enumerate(unique_labels):
            data_subset = np.transpose(plot_data[lab == l])
            if remove_outliers:
                data_subset[~filter[lab == l].T] = np.nan
            # ax.scatter(*data_subset, s=3e3*(1/dat.shape[0]), label=l)
            scatter = ax.scatter(*data_subset, label=l, edgecolors='black', color=colors[ix], s=size)
        fig = plt.gcf()
        # if i == 1 and legend:
        #     fig.legend(scatter, labels=unique_labels, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncols=7)
        if names is not None:
            ax.set_title(names[i])
        ax.set_xlabel(f'{method_names[method]}-1')
        ax.set_ylabel(f'{method_names[method]}-2')
        if n_components == 2 and square:
            ax.set_aspect('equal')
        elif n_components == 3:
            # ax.set_zlabel(f'{method_names[method]}-3')
            if square:
                # https://stackoverflow.com/a/13701747
                X, Y, Z = np.transpose(plot_data)
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
                for xb, yb, zb in zip(Xb, Yb, Zb):
                    ax.plot([xb], [yb], [zb], 'w')
                    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
    if not separate_dim:
        axs_xlim = np.array([ax.get_xlim() for ax in axs])
        axs_ylim = np.array([ax.get_ylim() for ax in axs])
        new_xlim = (axs_xlim.min(axis=0)[0], axs_xlim.max(axis=0)[1])
        new_ylim = (axs_ylim.min(axis=0)[0], axs_ylim.max(axis=0)[1])
        for ax in axs:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
    
    if i == 1 and legend:
        fig.legend(scatter, labels=unique_labels, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncols=7)
   
          
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    
    if save is not None:
        plt.savefig(save, format=fmt, dpi=180, bbox_inches='tight')
    
    if reduce_only:
        plt.close()
    else:
        plt.show()
        
    return datax