from utils import generate_color_map, return_generated_figure, get_legend_box, save_as_pdf_hbox, generate_dataset, save_as_pdf_allbox
from PIL import Image
from tqdm import tqdm
from itertools import product
from pathlib import Path
import pickle

import sys

dataset_list = [
    "ADNI_FOR_ALIGNED_TIME_SERIES",
    "PPMI_FOR_ALIGNED_TIME_SERIES",
    "AGECOHORT-PPMI-ADNI",
    "AllN2_BioReactor_2D_3D_Report_protein",
    "AllN2_Proteomics_BioReactor_2D_3D_Report_protein",
    "ALVEOLAR_metacelltype",
    "COVID19Proteomics",
    "ProteinCOVID19Proteomics",
    "FULLMIMIC_FEWFEATURES"

]

input_dataset_name = sys.argv[1]
data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")

fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
with open(fpath, 'rb') as handle:
    results_data = pickle.load(handle)

data = results_data['complete_dataframe'].copy()

alpha = 1

data = data[data['input_parameter_id'] == 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=03;min_dist=0.01;num_cores=16']
import sys; sys.exit()
MEAN = data.groupby(['Subtypes', 'z']).mean()
STD = data.groupby(['Subtypes', 'z']).std()
subject_distance_mapping_ = {}
g = list(data.groupby('subject_id'))
for i in tqdm(range(len(g))):
    temp = g[i][1].groupby(['Subtypes', 'z']).agg('mean')
    subject_distance_mapping[g[i][0]] = (temp - MEAN.loc[temp.index]).abs().mean().mean()


from tslearn.metrics import dtw
dtw_score_x = dtw(temp['x'], MEAN.loc[temp.index]['x'])
dtw_score_y = dtw(temp['y'], MEAN.loc[temp.index]['y'])

mean_val = np.mean(list(subject_distance_mapping.values()))
std_val = np.std(list(subject_distance_mapping.values()))

subject_id_map = {}
for key, value in subject_distance_mapping.items():
    if  value > (mean_val - 2 * std_val) and value < (mean_val + 2 * std_val):
        subject_id_map[key] = alpha
    else:
        subject_id_map[key] = alpha / 10



l = []
for i in range(len(data)):
    x = data.iloc[i]['x']
    y = data.iloc[i]['y']
    z = data.iloc[i]['z']
    subtype = data.iloc[i]['Subtypes']
    if  (MEAN.loc[(subtype, z)]['x'] - 2*MEAN.loc[(subtype, z)]['x']) < x and x < (MEAN.loc[(subtype, z)]['x'] + 2*MEAN.loc[(subtype, z)]['x']):
        l.append(alpha)
    else:
        l.append(alpha / 10)
