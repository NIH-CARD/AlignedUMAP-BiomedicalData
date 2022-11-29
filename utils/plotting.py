import pandas as pd
import numpy as np
import os
import umap
import pickle
from tqdm import tqdm
from datetime import datetime
import scipy.interpolate
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns


INTERP_KIND = {2:"linear", 3:"quadratic", 4:"cubic"}
def interpolate_paths(z, x, y, rep_id):
    consecutive_year_blocks = np.where(np.diff(z) != 1)[0] + 1
    z_blocks = np.split(z, consecutive_year_blocks)
    x_blocks = np.split(x, consecutive_year_blocks)
    y_blocks = np.split(y, consecutive_year_blocks)
    paths = []
    for block_idx, zs in enumerate(z_blocks):
        if len(zs) > 1:
            kind = INTERP_KIND.get(len(zs), "cubic")
        else:
            rep_id_list = np.array([rep_id]*len(zs))
            paths.append(
                (zs, x_blocks[block_idx], y_blocks[block_idx], rep_id_list)
            )
            continue
        # z = np.linspace(np.min(zs), np.max(zs), 100)
        z = np.round(np.linspace(np.min(zs), np.max(zs), 20), 2)
        x = scipy.interpolate.interp1d(zs, x_blocks[block_idx], kind=kind)(z)
        y = scipy.interpolate.interp1d(zs, y_blocks[block_idx], kind=kind)(z)
        rep_id_list = np.array([rep_id]*len(z))
        paths.append((z, x, y, rep_id_list))
    return paths


def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


def generate_animation(A, B, embeddings, color_column_name = 'subtype_category', dataset_name="PPMI_progression", feature_name="default_name", visualization_method='umap_aligned'):
    color_sequence = []
    color_codes = A[color_column_name]# .astype('category').cat.codes
    for enm, rep in enumerate(A.subject_id.unique()):
        color_index = color_codes.iloc[enm] 
        for enm_iter in range(len(B[enm])):
            bsize = B[enm][enm_iter][0].shape[0]
            B[enm][enm_iter] = list(B[enm][enm_iter])[1:]
            # B[enm][enm_iter].append(np.array([color_index]*bsize))
            # B[enm][enm_iter].append(np.array([rep]*bsize))
        color_sequence.append(color_index)
       
    interpolated_traces = B.copy()
    offsets = np.array(interpolated_traces).T
    offsets = offsets[:,:,0,:]
    offsets = np.transpose(offsets, (0,2,1))
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    scat = ax.scatter([], [], s=2)
    ax_bound = axis_bounds(np.vstack(embeddings))
    scat.set_array(np.array(pd.Series(color_sequence).astype('category').cat.codes))
    scat.set_cmap('Spectral')
    text = ax.text(ax_bound[0] + 0.5, ax_bound[2] + 0.5, '')
    ax.axis(ax_bound)
    ax.set(xticks=[], yticks=[])
    plt.tight_layout()

    # offsets = np.array(interpolated_traces).T
    num_frames = offsets.shape[0]
    def animate(i):
        scat.set_offsets(offsets[i])
        text.set_text(f'Frame {i}')
        return scat
    
    anim = animation.FuncAnimation(
        fig,
        init_func=None,
        func=animate,
        frames=num_frames,
        interval=40)
    os.makedirs(f"results_data/{dataset_name}/{visualization_method}/generated_plots", exist_ok=True)
    anim.save(f"results_data/{dataset_name}/{visualization_method}/generated_plots/{feature_name}_{color_column_name}.gif", writer="pillow")
    plt.close(anim._fig)


def plot_multidr(data, color_column_name, feature_name, dataset_name, visualization_method):
    z_cols = pd.unique(data['z'])
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 18))
    axlist = axs.reshape(-1)
    for e, z_col in enumerate(z_cols):
        print ('='*50)
        print (z_col)
        if z_col in ['Z_n_dt', 'Z_n_td', 0]:
            coloring_column = color_column_name
        else:
            coloring_column = 'category' 
        temp = data[data['z'] == z_col]
        if len(temp[coloring_column].unique()) > 100:
            sns.scatterplot(data=temp, x='x', y='y', ax=axlist[e])
            axlist[e].set_title(z_col)
            continue

        sns.scatterplot(data=temp, x='x', y='y', hue=coloring_column, ax=axlist[e])
        axlist[e].set_title(z_col)
        
        if len(temp[coloring_column].unique()) > 20:
            axlist[e].legend([],[], frameon=False)

    os.makedirs(f"results_data/{dataset_name}/{visualization_method}/generated_plots", exist_ok=True)
    plt.savefig(f"results_data/{dataset_name}/{visualization_method}/generated_plots/{feature_name}_{color_column_name}.pdf")
