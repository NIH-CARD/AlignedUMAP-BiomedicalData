from utils import generate_color_map, return_generated_figure, get_legend_box, save_as_pdf_hbox, generate_dataset, save_as_pdf_allbox
from PIL import Image
from tqdm import tqdm
from itertools import product
from pathlib import Path
import pickle

import sys

best_parameters = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'ADNI_FOR_ALIGNED_TIME_SERIES': 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'ALVEOLAR_metacelltype': 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'COVID19Proteomics': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', #DONE
    'NORM-COVID19Proteomics': 'metric=cosine;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', #DONE
    'NORM-ALVEOLAR_metacelltype': 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', #DONE
    'IMAGENORM-PPMI-ADNI': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', #DONE
    'MINMAX_MIMIC_ALLSAMPLES': 'metric=cosine;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.01;num_cores=16',# DONE
    'AllN2_BioReactor_2D_3D_Report_protein': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=03;min_dist=0.01;num_cores=16',# DONE
}

data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")
all_figs = []
row_titles = []
all_legends = []
input_dataset_name = sys.argv[1]# "ADNI_FOR_ALIGNED_TIME_SERIES"

params = [-1.25, 0, 1.25]  # should be zoomed out
eye_view = product(params, params, params)
eye_view = list(set(list(eye_view)))

fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
with open(fpath, 'rb') as handle:
    results_data = pickle.load(handle)

for value in tqdm(eye_view):
    if input_dataset_name == 'PPMI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['PD_h', 'HC']
        alpha = 1
    elif input_dataset_name == 'ADNI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['Dementia', 'Control']
        alpha = 1
    elif input_dataset_name == 'ALVEOLAR_metacelltype':
        colorname = 'metacelltype'
        num_subjects = 0.6
        selected_cats = ['T_cells', 'endothelial_cells', 'mesothelia_cells']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'NORM-ALVEOLAR_metacelltype':
        colorname = 'metacelltype'
        num_subjects = 0.6
        selected_cats = ['T_cells', 'endothelial_cells', 'mesothelia_cells']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'IMAGENORM-PPMI-ADNI':
        colorname = 'DIAGNOSIS'
        num_subjects = 0.6
        selected_cats = ['Dementia', 'Control']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'COVID19Proteomics':
        colorname = 'Acuity_max'
        num_subjects = 1
        selected_cats = [1, 2, 3, 4, 5]
        alpha = 1

    elif input_dataset_name == 'NORM-COVID19Proteomics':
        colorname = 'Acuity_max'
        num_subjects = 1
        selected_cats = [1, 2, 3, 4, 5]
        alpha = 1

    fig, (color_maps, nocolor_maps) = generate_dataset(results_data, input_dataset_name,  best_parameters[input_dataset_name], value , num_subjects, selected_cats, colorname, scale=1, alpha=alpha)
    all_figs.append(fig)
    all_legends.append(Image.open(get_legend_box(nocolor_maps, selected_cats, scale=1)))
    row_titles.append(f'eye{value}')

save_as_pdf_allbox(all_figs, row_titles, f'../camera_angles/{input_dataset_name}_camera.pdf', ncols=3, scale=1, all_legends=all_legends)
save_as_pdf_allbox(all_figs, row_titles, f'../camera_angles/{input_dataset_name}_camera_nolegend.pdf', ncols=3, scale=1, all_legends=None)
