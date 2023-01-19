import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
from itertools import product
import copy
import pickle
from pathlib import Path
from omegaconf import OmegaConf

from models.umap.main import get_embedding
config = OmegaConf.load(sys.argv[1])
num_cores = config['num_cores']
if num_cores == -1:
    num_cores = os.cpu_count()
data_dir = Path(config['data_dir'])
cache_dir = Path(config['cache_dir'])
result_dir = Path(config['result_dir'])
perform_interpolation = int(config['perform_interpolation'])
visualization_method = "umap"
dataset_name = config['dataset_name'].split('.csv')[0]
metadata_name = config['metadata_name'].split('.csv')[0]
if len(sys.argv) > 2:
    hyperparameter_index =  int(sys.argv[2])
else:
    hyperparameter_index = 0
sample_fraction = config['sample_fraction']
data = pd.read_csv(data_dir / f"{dataset_name}.csv")
data['subject_id'] = data['subject_id'].map(lambda x: str(x))
data['time_id'] = data['time_id'].map(lambda x: str(x))
if not sample_fraction == 1:
    sample_fraction = str(float(sample_fraction))
    all_subjects = list(data.subject_id.unique())
    selected_subjects = list(all_subjects[:int(len(all_subjects) * float(sample_fraction))])
    sample_text = f"_{sample_fraction}"
else:
    sample_fraction = "1.0"
    selected_subjects = list(data.subject_id.unique())
    sample_text = f"_{sample_fraction}"
data = data[data['subject_id'].isin(selected_subjects)]
data = data.set_index(['subject_id', 'time_id'])
if metadata_name == "":
    metadata = pd.DataFrame({'subject_id':data.reset_index()['subject_id'].unique()})
    metadata['color'] = 'NoColor'
else:
    metadata = pd.read_csv(data_dir / f'{metadata_name}.csv')

metadata = metadata.set_index('subject_id')
metadata.index = metadata.index.map(str)
metadata = metadata.to_dict()
umap_feature_variants = {}
umap_feature_variants[dataset_name] = np.array([True] * len(data.columns))
results_data = {}
output_df = []
metric_list = config['metric']
n_neighbors = config['n_neighbors']
min_dist = config['min_dist']
input_parameters = []
op = 0
for neig, dist, metric in list(product(n_neighbors, min_dist, metric_list)):
        op += 1
        temp = {}
        temp['metric'] = metric
        temp['n_neighbors'] = neig
        temp['min_dist'] = dist
        temp['num_cores'] = num_cores
        temp['sample_fraction'] = sample_fraction
        temp['id'] = ';'.join([f"{i}={j}" for i,j in temp.items()])
        temp['n_neighbors'] = int(neig)
        temp['min_dist'] = float(dist)
        input_parameters.append(copy.deepcopy(temp))
if hyperparameter_index < len(input_parameters):
    input_parameters = input_parameters[hyperparameter_index:hyperparameter_index + 1]
else:
    pass
print ('-'*50, "Executing", '-'*50)
print ('Dataset filename:', dataset_name, '|', 'Metadata filename:', metadata_name)
print ('Number of hyper parameters', len(input_parameters))
for feature_name, selected_feature in tqdm(umap_feature_variants.items()):
    for e_inp, input_parameter in enumerate(input_parameters):
        print('-' * 50, f'Hyper parameter (index={e_inp})', '-' * 50)
        print(input_parameter)
        print('-' * 50, '-' * 50)
        embeddings, time_taken = get_embedding(data, selected_feature, feature_name, data_dir=cache_dir, input_parameter=input_parameter)
        df = embeddings.copy()
        df = df.reset_index()
        df['feature_group'] = [feature_name] * len(df)
        for key, val in metadata.items():
            if key in ['subject_id', 'time_id', 'x', 'y', 'feature_group']:
                continue
            df[key] = df['subject_id'].map(lambda x: val.get(str(x), 'UNK'))
        df['input_parameter_id'] = [input_parameter['id']] * len(df) 
        output_df.append(df)
        results_data[f"{feature_name}-{input_parameter['id']}"] = {
            'data': df.copy(),
            'embeddings': copy.deepcopy(embeddings),
            'time_taken': time_taken,
        }
if hyperparameter_index == len(input_parameters):
    output_df = pd.concat(output_df, axis=0)
    results_data['complete_dataframe'] = output_df
    results_data['sample_size'] = len(selected_subjects)
    results_data['feature_size'] = data.shape[0]
    results_data['time_sequence'] = data.reset_index()['time_id'].unique().shape[0]
    os.makedirs(result_dir / f"{dataset_name}/{visualization_method}/generated_data", exist_ok=True)
    with open(result_dir / f"{dataset_name}/{visualization_method}/generated_data/{dataset_name}_{num_cores}{sample_text}.pickle", 'wb') as handle:
        pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50, '-' * 50)
    print("Congratulations!")
    print("Summarized results for all hyper parameters:", result_dir / f"{dataset_name}/{visualization_method}/generated_data/{dataset_name}_{num_cores}{sample_text}.pickle")