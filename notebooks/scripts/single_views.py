from utils import generate_color_map, return_generated_figure, get_legend_box, save_as_pdf_allbox, generate_dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import pickle

data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")

combined_best_parameters = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16'],
    # 'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
    'ADNI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
    # 'ALVEOLAR_metacelltype':  [[0, -1.25, -1.25], 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.10;num_cores=16'],
    'COVID19Proteomics': [[1.25, -1.25, -1.25], 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=05;min_dist=0.01;num_cores=16'],
}


ALLPLOTS = {
    'R0': {
            'C0': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap', 'metric=euclidean;n_neighbors=10;min_dist=0.01;num_cores=16'],
            'C1': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0, -1.25, -1.25] ],
            'C2': ['PPMI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [-1.25, 1.25, 1.25] ],
    },
    'R1': {
            'C0': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap', 'metric=euclidean;n_neighbors=10;min_dist=0.01;num_cores=16'],
            'C1': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [0.1, -2, -2] ],
            'C2': ['ADNI_FOR_ALIGNED_TIME_SERIES', 'umap_aligned', 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', [-1.75, 1.75, 1.75] ],
    },


}

all_figs = []
all_titles = []
all_legends = []
for input_dataset_name, value in tqdm(combined_best_parameters.items()):
    fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
    with open(fpath, 'rb') as handle:
        results_data = pickle.load(handle)
    mapping_legend = {}
    if input_dataset_name == 'PPMI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['PD_h', 'HC']
        mapping_legend = {'PD_h': 'High Parkinson Disease', 'HC': 'Healthy'}
        alpha = 1
    elif input_dataset_name == 'ADNI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['Dementia', 'Control']
        mapping_legend = {'Control': 'Healthy', 'Dementia': 'Dementia'}
        alpha = 1
    elif input_dataset_name == 'ALVEOLAR_metacelltype':
        colorname = 'metacelltype'
        num_subjects = 0.6
        selected_cats = ['T_cells', 'endothelial_cells', 'mesothelia_cells']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
    elif input_dataset_name == 'COVID19Proteomics':
        colorname = 'Acuity_max'
        mapping_legend = {1: 'Deceased', 2: 'Ventilated', 3: 'Hospitalized (O2)', 4: 'Hospitalized', 5: 'Recovered'}
        num_subjects = 1
        selected_cats = [2, 3]
        alpha = 1
    all_titles.append(input_dataset_name)
    fig, (color_maps, nocolor_maps) = generate_dataset(results_data, input_dataset_name, value[1], value[0], num_subjects, selected_cats, colorname, scale=2, alpha=alpha, transparency_effect=True)
    all_figs.append(fig)
    all_legends.append(Image.open(get_legend_box(nocolor_maps, selected_cats, scale=2, mapping_legend=mapping_legend)))

save_as_pdf_allbox(all_figs , all_titles , f'../final_images/SingleView_alv.pdf', ncols=3, scale=2, all_legends=all_legends)