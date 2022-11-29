import pandas as pd
import numpy as np
import os
import umap
import pickle
from pathlib import Path
from numba import njit, prange, set_num_threads


from tqdm import tqdm
from datetime import datetime
import scipy.interpolate
import copy

# data_dir = Path ("/data/dadua2/projects.link/baseTimeVaryingAlignedUMAP")
# data_dir = Path ("/data/CARD/projects/ImagingBasedProgressionGWAS/ECG")
def get_embedding(data, selected_feature_index, feature_group_name, data_dir, input_parameter={
        "n_neighbors": 15,
        "min_dist": 0.1,
        'id': 'parameters_id_name',
        'metric': 'euclidean',
        'num_cores': '32',
    }):


    os.makedirs ( data_dir / f"cache_data/{feature_group_name}", exist_ok=True)
    if os.path.exists ( data_dir / f"cache_data/{feature_group_name}/{input_parameter['id']}_umap.pickle"):
        print ("Embedding Exists:", data_dir / f"cache_data/{feature_group_name}/{input_parameter['id']}_umap.pickle")
        with open( data_dir / f"cache_data/{feature_group_name}/{input_parameter['id']}_umap.pickle", 'rb') as handle:
            data = pickle.load(handle)
        embeddings = data['embeddings']
        return embeddings, data['total_time_taken']

    set_num_threads(int(input_parameter["num_cores"]))
    print ("Started Working on Aligned UMAP")

    T1_data = data.iloc[:, selected_feature_index].dropna().reset_index()
    ####################################################################################################################
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

    print ('Completed At', datetime.now())
    data = {
        'embeddings': pd.DataFrame(embeddings, columns=['x', 'y'], index=T1_data.set_index(['subject_id', 'time_id']).index),
	    'total_time_taken': str(datetime.now() - start_time)
    }
    with open( data_dir / f"cache_data/{feature_group_name}/{input_parameter['id']}_umap.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data['embeddings'], data['total_time_taken']


# if __name__ == "__main__":
#     # Create V and M as the dataset
# 
#     data1 = pd.read_csv('singlecelltimevariant/DIA_Master_Headings_AQ.csv')
#     X = data1.drop(columns=['PG.ProteinGroups', 'PG.ProteinAccessions', 'PG.Genes', 'PG.ProteinDescriptions'])
#     protein_index_group = dict(zip(list(data1.index), list(data1['PG.ProteinGroups'])))
#     Z = X.T.reset_index().rename(columns={'index':'variable'})
#     Z['variable'] = Z['variable'].map(lambda x: x.replace('iPSC', 'iPSC_D3') if 'iPSC' in x else x)
#     Z['time_id'] = Z['variable'].map(lambda x: int(x.split(' ')[0].split('_')[1].split('D')[-1])) 
#     Z['time_id'] = Z['time_id'].map(lambda x: f'D{x:03}')
#     Z['subject_id'] = Z['variable'].map(lambda x: x.split('_')[0] + '_' + x.split(' ')[-1])
#     Z = Z.drop(columns=['variable'])
# 
#     row_index_arrays = [[f"G{i+1}" for i in range(len(Z))], list(Z['subject_id'])]
#     row_index_map = pd.MultiIndex.from_arrays(row_index_arrays, names=('group_id', 'subject_id'))
#     Z = Z.set_index(['subject_id', 'time_id'])
#     columns_list = [col for col in Z.columns ]
#     
#     
# 
#     
#     Z = Z.replace('Filtered', np.nan)
#     Z = Z.astype(float)
#     Z = Z.apply(lambda x: x.fillna(x.median()),axis=0)
# 
#    
#     
# 
#     data3 = pd.read_excel("singlecelltimevariant/Markers_General.xlsx")
#     markers_general = {}
#     for i in range(0, len(data3.columns), 2):
#         temp = data3.iloc[:, [i, i+1]].dropna()
#         markers_general[data3.columns[i]]  = dict(zip(list(temp.iloc[:, 1]), list(temp.iloc[:, 0]))) 
# 
#     data4 = pd.read_excel("singlecelltimevariant/Markers_Specific.xlsx")
#     markers_specific = {}
#     for i in range(0, len(data4.columns), 2):
#         temp = data4.iloc[:, [i, i+1]].dropna()
#         markers_specific[data4.columns[i]]  = dict(zip(list(temp.iloc[:, 1]), list(temp.iloc[:, 0]))) 
#     
#     group_name_mapping = {}
#     for i,j in markers_general.items():
#         for k, m in j.items():
#             if not k in group_name_mapping:
#                 group_name_mapping[k] = {}
#             group_name_mapping[k]['name'] = m
#             group_name_mapping[k]['group_name'] = f"General-{i.split('/')[0]}"
#             
#     for i,j in markers_specific.items():
#         for k, m in j.items():
#             if not k in group_name_mapping:
#                 group_name_mapping[k] = {}
#             group_name_mapping[k]['name'] = m
#             group_name_mapping[k]['group_name'] = f"Specific-{i.split('/')[0]}"
#     
#     column_index_arrays = []
#     column_index_names = []
#     feature_ids = pd.Series(list(map(lambda x: protein_index_group[x], list(columns_list))))
#     column_index_arrays.append(list(feature_ids))
#     column_index_names.append('feature_id')
#     get_feature_group = lambda x: 'Other' if group_name_mapping.get(x, None) is None else group_name_mapping[x]['group_name']
#     column_index_arrays.append(feature_ids.map(get_feature_group))
#     column_index_names.append('group_name') 
#     col_index_map = pd.MultiIndex.from_arrays(column_index_arrays, names=column_index_names)
#     Z.columns = col_index_map
#     
#     if False:
#         group_name_patterns = ['Specific', 'General', 'Cholinergic', 'Inhibitory', 'Astrocyte', 'Mature Neuron', 'Precursor', 'Glutamatergic', 'Synaptic', 'Progenitor']
#         umap_variants_type = {}
#         for group_name_pattern in group_name_patterns:
#             umap_variants_type[group_name_pattern] = Z.columns.get_level_values('group_name').str.contains(group_name_pattern)
#     
#         umap_variants_type['All'] = np.array([True] * len(Z.columns)) 
# 
#         results_data = {}
#         output_df = []
#         for feature_name, selected_feature in tqdm(umap_variants_type.items()):
#             print ('='*10, feature_name)
#             embeddings, V, T1_data, age_list = get_embedding(Z, selected_feature, feature_name)
#             df = pd.DataFrame(np.vstack(embeddings), columns=('x', 'y'))
#             df['z'] = np.concatenate([[year] * len(embeddings[i]) for i, year in enumerate(range(len(age_list)))])
#             df['subject_id'] = np.concatenate([v.index for v in V])
#             df['feature_group'] = [feature_name] * len(df)
#             df['cell_category'] = df['subject_id'].map(lambda x: x.split('_')[0])
#             df['cell_replication'] = df['subject_id'].map(lambda x: int(x.split('_')[1]))
# 
#             trace_inputs = []
#             for rep in tqdm(df.subject_id.unique()):
#                 z = df.z[df.subject_id == rep].values
#                 x = df.x[df.subject_id == rep].values
#                 y = df.y[df.subject_id == rep].values
#                 trace_inputs.append(interpolate_paths(z, x, y, rep))
# 
#             results_data[feature_name] = {
#                 'data': df,
#                 'trace_inputs': trace_inputs,
#                 
#             }
#             output_df.append(df) 
# 
#         output_df = pd.concat(output_df, axis=0)
#         results_data['complete_dataframe'] = output_df
#         results_data['age_list'] = age_list 
#         with open(f'results_data/umap_embeddings_traces.pickle', 'wb') as handle:
#             pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
# 
#     if True:
#          # perform transpose
#         A = Z.unstack().T.reset_index()
#         drop_columns = ['iPSC_1', 'iPSC_2', 'iPSC_3', 'iPSC_4', 'iPSC_5', 'iPSC_6']
#         D = A.drop(columns=drop_columns)
#         L = D.set_index(['feature_id', 'group_name', 'time_id'])
#         F = L.apply(lambda x: x.fillna(x.median()),axis=1)
#         P = F.reset_index().rename(columns={'feature_id': 'subject_id'})
#         feature_id_mapping = dict(zip(list(P['subject_id']), list(P['group_name'])))
#         P = P.set_index(['subject_id', 'time_id'])
#         P = P.drop(columns=['group_name'])
#         umap_variants_type = {}
#         umap_variants_type['transpose_All'] = np.array([True] * len(P.columns)) 
#     
#         results_data = {}
#         output_df = []
#         for feature_name, selected_feature in tqdm(umap_variants_type.items()):
#             print ('='*10, feature_name)
#             embeddings, V, T1_data, age_list = get_embedding(P, selected_feature, feature_name)
#             df = pd.DataFrame(np.vstack(embeddings), columns=('x', 'y'))
#             df['z'] = np.concatenate([[year] * len(embeddings[i]) for i, year in enumerate(range(len(age_list)))])
#             df['subject_id'] = np.concatenate([v.index for v in V])
#             df['feature_group'] = [feature_name] * len(df)
#             df['protein_category'] = df['subject_id'].map(lambda x: feature_id_mapping.get(x, 'Other'))
#     
#             trace_inputs = []
#             for rep in tqdm(df.subject_id.unique()):
#                 z = df.z[df.subject_id == rep].values
#                 x = df.x[df.subject_id == rep].values
#                 y = df.y[df.subject_id == rep].values
#                 trace_inputs.append(interpolate_paths(z, x, y, rep))
#     
#             results_data[feature_name] = {
#                 'data': df,
#                 'trace_inputs': trace_inputs,
#             }
#             output_df.append(df) 
#     
#         output_df = pd.concat(output_df, axis=0)
#         results_data['complete_dataframe'] = output_df
#         results_data['age_list'] = age_list 
#         with open(f'results_data/transpose_umap_embeddings_traces.pickle', 'wb') as handle:
#             pickle.dump(results_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 
# 
# 
#     # Z.loc[Z.index.get_level_values('subject_id').str.contains('SMAD')]
#     # Z.index.get_locs([slice(None), Z.index.get_level_values('subject_id')])
#        
#     
# 
# 
# 
# 
#     # V = ?
#     # M = ?
# 
#     print ("Hello")
# 
# # df = pd.DataFrame(np.vstack(embeddings), columns=('x', 'y'))
# # df['z'] = np.concatenate([[year] * len(embeddings[i]) for i, year in enumerate(range(len(age_list)))])
# # df['subject_id'] = np.concatenate([v.index for v in V])
# # df = pd.merge(df, metadata, left_index=True, right_index=True)