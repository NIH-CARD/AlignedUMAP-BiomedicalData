from utils import generate_color_map, save_as_pdf_allbox, return_generated_figure, get_legend_box, save_as_pdf_hbox, generate_dataset
from PIL import Image
from tqdm import tqdm
from itertools import product
from pathlib import Path
import pandas as pd
import pickle
import sys

data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")

all_figs = []
row_titles = []
all_legends = []
input_dataset_name = sys.argv[1]# "ADNI_FOR_ALIGNED_TIME_SERIES"
eye_view = [1.25, -1.25, -1.25]

fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
with open(fpath, 'rb') as handle:
    results_data = pickle.load(handle)

for value in tqdm(list(results_data['complete_dataframe']['input_parameter_id'].unique())):

    mapping_legend = {}
    if input_dataset_name == 'ADNI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['Dementia', 'Control']
        alpha = 1
    elif input_dataset_name == 'ALVEOLAR_metacelltype':
        colorname = 'metacelltype'
        num_subjects = 0.6
        selected_cats = ['T_cells', 'endothelial_cells', 'mesothelia_cells']
        alpha = 0.8
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'NORM-ALVEOLAR_metacelltype':
        colorname = 'metacelltype'
        num_subjects = 0.6
        selected_cats = ['T_cells', 'endothelial_cells', 'mesothelia_cells']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]


    elif input_dataset_name == 'AGECOHORT-PPMI-ADNI':
        colorname = 'DIAGNOSIS'
        num_subjects = 0.3
        selected_cats = ['Dementia', 'Control']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'IMAGENORM-PPMI-ADNI':
        colorname = 'DIAGNOSIS'
        num_subjects = 0.4
        selected_cats = ['Dementia', 'Control']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'RelationIMAGENORM-PPMI-ADNI':
        colorname = 'DIAGNOSIS'
        num_subjects = 0.4
        selected_cats = ['Dementia', 'Control']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]


    elif input_dataset_name == 'AllN2_BioReactor_2D_3D_Report_protein':
        colorname = 'label'
        num_subjects = 1
        selected_cats = ['MICU', 'CSRU']
        selected_cats = []
        alpha = 1

    elif input_dataset_name == 'AllN2_Proteomics_BioReactor_2D_3D_Report_protein':
        colorname = 'label'
        num_subjects = 0.1
        selected_cats = []
        alpha = 1

    elif input_dataset_name == 'NORM-AllN2_Proteomics_BioReactor_2D_3D_Report_protein':
        colorname = 'label'
        num_subjects = 0.05
        selected_cats = []
        alpha = 1

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

    elif input_dataset_name == 'MINMAX_MIMIC_SHORTSAMPLES':
        colorname = 'y_true'
        num_subjects = 0.6
        selected_cats = []
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER']=='M']
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['z']<=4]
        results_data['complete_dataframe'] = results_data['complete_dataframe'].drop_duplicates(subset=['z', 'subject_id', 'input_parameter_id'])

    elif input_dataset_name == 'NORM_MIMIC_ALLSAMPLES':
        colorname = 'LAST_CAREUNIT'
        num_subjects = 0.1
        selected_cats = ['MICU', 'CSRU']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER']=='M']
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['z']<=4]
        results_data['complete_dataframe'] = results_data['complete_dataframe'].drop_duplicates(subset=['z', 'subject_id', 'input_parameter_id'])

    elif input_dataset_name == 'MINMAX_MIMIC_ALLSAMPLES':
        colorname = 'LAST_CAREUNIT'
        num_subjects = 0.1
        selected_cats = ['MICU', 'CSRU']
        alpha = 1
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER']=='M']
        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['z']<=4]
        results_data['complete_dataframe'] = results_data['complete_dataframe'].drop_duplicates(subset=['z', 'subject_id', 'input_parameter_id'])


    elif input_dataset_name == 'FULLMIMIC_FEWFEATURES':
        colorname = 'labels'
        num_subjects = 0.05
        selected_cats = []
        alpha = 1
        results_data['complete_dataframe']['period_length'] = pd.qcut(results_data['complete_dataframe']['period_length'], 4, retbins=True, precision=0)[0].map(lambda i: "{:03}-{:03}".format((int(i.left)), (int(i.right))))
        # results_data['complete_dataframe']['period_length'] = pd.qcut(results_data['complete_dataframe']['period_length'], 4, retbins=True, precision=0)[0].map(lambda i: int(i.left))
        colorname = 'period_length'
        # results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

    elif input_dataset_name == 'PPMI_FOR_ALIGNED_TIME_SERIES':
        colorname = 'Subtypes'
        num_subjects = 1
        selected_cats = ['PD_h', 'HC']
        alpha = 1

    else:
        continue

    fig, (color_maps, nocolor_maps) = generate_dataset(results_data, input_dataset_name,  value, eye_view , num_subjects, selected_cats, colorname, scale=1, alpha=alpha, transparency_effect=True)
    all_figs.append(fig)
    all_legends.append(Image.open(get_legend_box(nocolor_maps, selected_cats, scale=1, mapping_legend=mapping_legend)))
    title = f"{':'.join([k.split('=')[-1] for k in value.split(';')])}"
    row_titles.append(title)

save_as_pdf_allbox(all_figs, row_titles, f'../parameter_views/{input_dataset_name}_parameters.pdf', ncols=4, scale=1, all_legends=all_legends)
save_as_pdf_allbox(all_figs, row_titles, f'../parameter_views/{input_dataset_name}_parameters_nolegend.pdf', ncols=4, scale=1, all_legends=None)
