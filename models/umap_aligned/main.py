import pandas as pd
import numpy as np
import os
import umap
import pickle
from numba import njit, prange, set_num_threads
from datetime import datetime
from datetime import timedelta
import copy

def get_embedding(data, selected_feature_index, feature_group_name, data_dir, input_parameter=None):
    if input_parameter is None:
        input_parameter = {
            "alignment_regularisation": 1.0e-2,
            "alignment_window_size": 3,
            "n_neighbors": 15,
            "min_dist": 0.1,
            'id': 'parameters_id_name',
            'metric': 'euclidean',
            'num_cores': '32',
        }
    os.makedirs ( data_dir / f"{feature_group_name}", exist_ok=True)
    if os.path.exists ( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap_aligned.pickle"):
        print ("Embedding exists at:", data_dir / f"{feature_group_name}/{input_parameter['id']}_umap_aligned.pickle")
        with open( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap_aligned.pickle", 'rb') as handle:
            data = pickle.load(handle)
        embeddings = data['embeddings']
        V = data['V']
        T1_data = data['T1_data']
        age_list = data['age_list']
        return embeddings, V, T1_data, age_list, data['total_time_taken']
    set_num_threads(int(input_parameter["num_cores"]))
    print ("Aligned UMAP running...")
    T1_data = data.iloc[:, selected_feature_index].dropna().reset_index()
    age_list = sorted(list(T1_data['time_id'].unique()))
    if type(input_parameter['n_neighbors']) is list:
        n_neighbors = input_parameter['n_neighbors']
        min_dist = input_parameter['min_dist'] 
    else:
        n_neighbors = [input_parameter['n_neighbors'],] * len(age_list)
        min_dist = [input_parameter['min_dist'],] * len(age_list)
    V = []
    for age in age_list:
        temp = T1_data[T1_data['time_id'] == age] 
        V.append(temp.drop(columns=['time_id']).set_index('subject_id'))
    def make_relation(from_df, to_df):
        left = pd.DataFrame(data=np.arange(len(from_df)), index=from_df.index)
        right = pd.DataFrame(data=np.arange(len(to_df)), index=to_df.index)
        merge = pd.merge(left, right, left_index=True, right_index=True)
        return dict(merge.values)
    relations = [make_relation(x,y) for x, y in zip(V[:-1], V[1:])]
    V_copy = copy.deepcopy(V)
    print ('Starts at:', datetime.now().replace(microsecond=0))
    start_time = datetime.now()
    aligned_mapper = umap.AlignedUMAP(
        metric=input_parameter['metric'],
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        alignment_regularisation=input_parameter['alignment_regularisation'],
        alignment_window_size=input_parameter['alignment_window_size'],
        # n_jobs=int(input_parameter['num_cores']),
        n_epochs=50,
        random_state=42,
    ).fit(V, relations=relations)
    print ('Completed at:', datetime.now().replace(microsecond=0))
    td = timedelta(seconds=round((datetime.now() - start_time).total_seconds()))
    print ('Total time taken (hh:mm:ss):', str(td))
    embeddings = aligned_mapper.embeddings_
    data = {
        'embeddings': list(embeddings),
        'V': [pd.DataFrame(index=v.index, columns=v.columns) for v in V_copy],
        'T1_data': "",# T1_data,
        'age_list': age_list,
        'total_time_taken': str(datetime.now() - start_time)
    }
    with open( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap_aligned.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    return list(embeddings), V_copy, T1_data, age_list, data['total_time_taken']