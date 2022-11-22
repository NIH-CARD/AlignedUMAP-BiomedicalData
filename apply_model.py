# from numba import njit, prange, set_num_threads

import pandas as pd
import copy
import numpy as np
import pandas as pd
import numpy as np
import os
import umap
from tqdm import tqdm
from datetime import datetime
import scipy.interpolate
import copy
from collections import defaultdict
import sys
from itertools import product
import copy
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from utils.plotting import generate_animation 
from utils.plotting import interpolate_paths 
from utils.plotting import plot_multidr 

num_cores = str(sys.argv[5])
# set_num_threads(int(num_cores))
config_fname = sys.argv[4]
config = OmegaConf.load(config_fname)
data_dir = Path(config['data_dir']) # Path ("/data/dadua2/projects.link/baseTimeVaryingAlignedUMAP")
cache_dir = Path(config['cache_dir']) # Path ("/data/dadua2/projects.link/baseTimeVaryingAlignedUMAP")
result_dir = Path(config['result_dir'])
# method to apply
perform_interpolation = int(config['perform_interpolation'])

# gifs_color_column_names = [ 'Subtypes' ]  
# gifs_color_column_names = [ 'stand_class' ]  
# gifs_color_column_names = [ 'heartbeat_class' ]  
# gifs_color_column_names = [ 'natops_class' ]  
# gifs_color_column_names = [ 'diagnostic_superclass' ]  
# visualization_method = 'tSNE_dynamic', 'umap', 'umap_aligned', 'multidr'
# visualization_method = sys.argv[1]
visualization_method = "umap_aligned"
# dataset_name = "PPMI_BRAINSTEP_MRI_T1", "ADNI_FOR_ALIGNED_TIME_SERIES", "PPMI_FOR_ALIGNED_TIME_SERIES"
dataset_name = sys.argv[1]
print (dataset_name)
data = pd.read_hdf(data_dir / f"dataset/{dataset_name}.h5", key='data')
data = data.reset_index()

data['subject_id'] = data['subject_id'].map(lambda x: str(x))
data['time_id'] = data['time_id'].map(lambda x: str(x))
if len(sys.argv) >= 7: # "sample_fraction" in config:
    sample_fraction = str(sys.argv[6])
    all_subjects = list(data.subject_id.unique())
    selected_subjects = list(all_subjects[:int(len(all_subjects) * float(sample_fraction))])
    sample_text = f"_{sample_fraction}"
else:
    sample_fraction = "1.0"
    selected_subjects = list(data.subject_id.unique())
    sample_text = f"_{sample_fraction}"

data = data[data['subject_id'].isin(selected_subjects)]
data = data.set_index(['subject_id', 'time_id'])


# metadata_name = "PPMI_labels"
# if len(sys.argv) > 2:
metadata_name = sys.argv[2]
metadata = pd.read_hdf(data_dir / f'metadata/{metadata_name}.h5', key='data')
metadata.index = metadata.index.map(str)
metadata = metadata.to_dict()


index_name = int(sys.argv[3])
# else:
#     metadata = {}
from models.umap_aligned.main import get_embedding
# feature selection
umap_feature_variants = {}
umap_feature_variants[dataset_name] = np.array([True] * len(data.columns)) 

results_data = {}
output_df = []

metric_list = config['metric']
alignment_regularisation = config['alignment_regularisation']# [0.003, 0.03]# [0.003, 0.03, 0.1]
alignment_window_size = config['alignment_window_size'] # [3, 6] # 1, 3
n_neighbors = config['n_neighbors']# [15, 25] #  5, 10
min_dist = config['min_dist']# [0.01, 0.2]
input_parameters = []
op = 0
for reg, ws in list(product(alignment_regularisation, alignment_window_size)):
    for neig, dist, metric in list(product(n_neighbors, min_dist, metric_list)):
        # if op >= 1:
        #     break
        op += 1
        temp = {}
        temp['metric'] = metric
        temp['alignment_regularisation'] = reg
        temp['alignment_window_size'] = ws
        temp['n_neighbors'] = neig
        temp['min_dist'] = dist
        temp['num_cores'] = num_cores
        temp['sample_fraction'] = sample_fraction
        temp['id'] = ';'.join([f"{i}={j}" for i,j in temp.items()])
        temp['alignment_regularisation'] = float(reg)
        temp['alignment_window_size'] = int(ws)
        temp['n_neighbors'] = int(neig)
        temp['min_dist'] = float(dist)
        temp['min_dist'] = float(dist)
        input_parameters.append(copy.deepcopy(temp))
# input_parameters = input_parameters[::-1]
if index_name < len(input_parameters):
    input_parameters = input_parameters[index_name:index_name + 1]
else:
    pass
print (input_parameters)

for feature_name, selected_feature in tqdm(umap_feature_variants.items()):
    print ('='*10, feature_name)
    for input_parameter in input_parameters:
        print ('*'*30)
        print (input_parameter)
        print ('*'*30)
        embeddings, V, T1_data, age_list, time_taken = get_embedding(data, selected_feature, feature_name, data_dir=cache_dir, input_parameter=input_parameter)
        df = pd.DataFrame(np.vstack(embeddings), columns=('x', 'y'))
        df['z'] = np.concatenate([[year] * len(embeddings[i]) for i, year in enumerate(range(len(age_list)))])
        df['subject_id'] = np.concatenate([v.index for v in V])
        df['feature_group'] = [feature_name] * len(df)
        for key, val in metadata.items():
            df[key] = df['subject_id'].map(lambda x: val.get(str(x), 'UNK')) 
        df['input_parameter_id'] = [input_parameter['id']] * len(df) 
        trace_inputs = []
        extended_interpolated_data = {}
        if perform_interpolation:
            for rep in tqdm(df.subject_id.unique()):
                z = df.z[df.subject_id == rep].values
                x = df.x[df.subject_id == rep].values
                y = df.y[df.subject_id == rep].values
                l = interpolate_paths(z, x, y, rep)
                trace_inputs.append(l)
            extended_interpolated_data = defaultdict(list)
            for trace_input in trace_inputs:
                for z, x, y, rep in trace_input:
                    extended_interpolated_data['x'].extend(x)
                    extended_interpolated_data['z'].extend(z)
                    extended_interpolated_data['y'].extend(y)
                    extended_interpolated_data['text'].extend(rep)
            extended_interpolated_data = pd.DataFrame(extended_interpolated_data)
            extended_interpolated_data['z'] = extended_interpolated_data['z'].map(lambda x: format(round(x, 2), '.2f') )
            extended_interpolated_data['z'] = extended_interpolated_data['z'].map(lambda x: f"z-{x}")
            extended_interpolated_data = extended_interpolated_data.rename(columns={'text': 'subject_id', 'z': 'time_id'})
            extended_interpolated_data['subject_id'] = extended_interpolated_data['subject_id'].map(str)
            extended_interpolated_data = extended_interpolated_data.set_index(['subject_id', 'time_id'])
        # else:
        output_df.append(df)
        results_data[f"{feature_name}-{input_parameter['id']}"] = {
            'data': df.copy(),
            'trace_inputs': copy.deepcopy(trace_inputs),
            'embeddings': copy.deepcopy(embeddings),
            'time_taken': time_taken,
            'extended_interpolated_data': extended_interpolated_data
        }

if index_name == len(input_parameters):
    output_df = pd.concat(output_df, axis=0)
    results_data['complete_dataframe'] = output_df
    results_data['sample_size'] = len(selected_subjects)
    results_data['feature_size'] = data.shape[0]
    results_data['time_sequence'] = data.reset_index()['time_id'].unique().shape[0]
    results_data['age_list'] = age_list
    os.makedirs(result_dir / f"results_data/{dataset_name}/{visualization_method}/generated_data", exist_ok=True)
    with open(result_dir / f"results_data/{dataset_name}/{visualization_method}/generated_data/{dataset_name}_{num_cores}{sample_text}.pickle", 'wb') as handle:
        pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)