import pandas as pd
import os
import umap
import pickle
from numba import njit, prange, set_num_threads
from datetime import datetime
from datetime import timedelta

def get_embedding(data, selected_feature_index, feature_group_name, data_dir, input_parameter=None):
    if input_parameter is None:
        input_parameter = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            'id': 'parameters_id_name',
            'metric': 'euclidean',
            'num_cores': '32',
        }
    os.makedirs ( data_dir / f"{feature_group_name}", exist_ok=True)
    if os.path.exists ( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap.pickle"):
        print ("Embedding exists at:", data_dir / f"{feature_group_name}/{input_parameter['id']}_umap.pickle")
        with open( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap.pickle", 'rb') as handle:
            data = pickle.load(handle)
        embeddings = data['embeddings']
        return embeddings, data['total_time_taken']
    set_num_threads(int(input_parameter["num_cores"]))
    print ("UMAP running...")
    T1_data = data.iloc[:, selected_feature_index].dropna().reset_index()
    n_neighbors = input_parameter['n_neighbors']
    min_dist = input_parameter['min_dist']
    print ('Starts At:', datetime.now())
    start_time = datetime.now()
    embeddings = umap.UMAP(
        metric=input_parameter['metric'],
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        # n_jobs=int(input_parameter['num_cores']),
        n_epochs=50,
        random_state=42,
    ).fit_transform(T1_data.set_index(['subject_id', 'time_id']))
    print('Completed at:', datetime.now().replace(microsecond=0))
    td = timedelta(seconds=round((datetime.now() - start_time).total_seconds()))
    print('Total time taken (hh:mm:ss):', str(td))
    data = {
        'embeddings': pd.DataFrame(embeddings, columns=['x', 'y'], index=T1_data.set_index(['subject_id', 'time_id']).index),
	    'total_time_taken': str(datetime.now() - start_time)
    }
    with open( data_dir / f"{feature_group_name}/{input_parameter['id']}_umap.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data['embeddings'], data['total_time_taken']