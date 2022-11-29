import os
from utils import generate_color_map, return_generated_figure, get_legend_box, save_as_pdf_hbox, generate_dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pickle
import sys
data_dir = Path("/Users/dadua2/projects")
# data_dir = Path("/data/CARD/projects/ImagingBasedProgressionGWAS/projects")

best_angles = {
    # 'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], [-1.25, 1.25, 1.25], [-1.25, -1.25, -1.25]],
    'PPMI_FOR_ALIGNED_TIME_SERIES': [ [0, -1.25, -1.25] , [-1.25, 1.25, 1.25],],
    # 'ADNI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25], [0, -1.25, -1.25], [-1.25, 1.25, 1.25]],
    'ADNI_FOR_ALIGNED_TIME_SERIES': [[0.1, -2, -2] , [-1.75, 1.75, 1.75]],
    # 'ADNI_FOR_ALIGNED_TIME_SERIES': [[0, -1.25, -1.25] , [-1.75, 1.75, 1.75]],
    # 'ADNI_FOR_ALIGNED_TIME_SERIES': [[0, -2, -2] , [-1.75, 1.75, 1.75]],
    'ALVEOLAR_metacelltype':  [[1.25, -1.25, -1.25], [0, -1.25, -1.25], [1.25, 1.25, 1.25]],  #, # [1.25, -1.25, -1.25]
    "COVID19Proteomics": [[1.25, -1.25, -1.25], [0, -1.25, 0], [-1.25, -1.25, -1.25]],
    "NORM-COVID19Proteomics": [[1.25, -1.25, -1.25], [0, 1.25, -1.25], [1.25, 1.25, -1.25]],
    "NORM-ALVEOLAR_metacelltype": [[0, -1.25, -1.25], [1.25, -1.25, -1.25], [-1.25, 1.25, -1.25]],
    "IMAGENORM-PPMI-ADNI": [[-1.25, 1.25, -1.25], [0, 1.25, -1.25], [-1.25, 1.25, 1.25]],
    'MINMAX_MIMIC_ALLSAMPLES': [[1.25, -1.25, -1.25], [-1.25, 1.25, 1.25], [-1.25, -1.25, -1.25]],
    'AllN2_BioReactor_2D_3D_Report_protein': [[1.25, -1.25, -1.25], [-1.25, 1.25, 1.25], [-1.25, -1.25, -1.25]],# DONE
}

best_parameters = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'ADNI_FOR_ALIGNED_TIME_SERIES': 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'ALVEOLAR_metacelltype': 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=10;min_dist=0.10;num_cores=16', #DONE
    'COVID19Proteomics': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', #DONE
    'NORM-COVID19Proteomics': 'metric=cosine;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', #DONE
    'NORM-ALVEOLAR_metacelltype': 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.01;num_cores=16', #DONE
    'IMAGENORM-PPMI-ADNI': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=10;min_dist=0.01;num_cores=16', #DONE
    'MINMAX_MIMIC_ALLSAMPLES': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.01;num_cores=16',# DONE
    'AllN2_BioReactor_2D_3D_Report_protein': 'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=2;n_neighbors=03;min_dist=0.01;num_cores=16',# DONE
}

name_mapping = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': 'PPMI Clinical Measurements',
    'ADNI_FOR_ALIGNED_TIME_SERIES': 'ADNI Clinical Measurements',
    'ALVEOLAR_metacelltype': 'Longitudinal scRNA Mice Lung',
    'NORM-ALVEOLAR_metacelltype': 'Longitudinal scRNA Mice Lung (Normalized)',
    'COVID19Proteomics': 'Longitudinal Proteomics COVID19',
    'NORM-COVID19Proteomics': 'Longitudinal Proteomics COVID19 (Normalized)',
    'IMAGENORM-PPMI-ADNI': 'ADNI+PPMI T1 MRI (Ageing)',
    'AllN2_BioReactor_2D_3D_Report_protein': 'Longitudinal Stem Cell Proteomics',
    "MINMAX_MIMIC_ALLSAMPLES": 'MIMIC-III ICU data (Normalized)'
}

color_name = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': 'Subtypes',
    'ADNI_FOR_ALIGNED_TIME_SERIES': 'Subtypes',
    'ALVEOLAR_metacelltype': 'cell.type',
    'NORM-ALVEOLAR_metacelltype': 'cell.type',
    'COVID19Proteomics': 'Longitudinal Proteomics COVID19',
    'NORM-COVID19Proteomics': 'Acuity_max',
    'IMAGENORM-PPMI-ADNI': 'GENDER_DIAGNOSIS',
    'AllN2_BioReactor_2D_3D_Report_protein': 'label',
    "MINMAX_MIMIC_ALLSAMPLES": 'LAST_CAREUNIT_MALE'
}

all_inputs = ['PPMI_FOR_ALIGNED_TIME_SERIES', 'ADNI_FOR_ALIGNED_TIME_SERIES', 'NORM-ALVEOLAR_metacelltype',  'NORM-COVID19Proteomics', 'IMAGENORM-PPMI-ADNI', 'AllN2_BioReactor_2D_3D_Report_protein', "MINMAX_MIMIC_ALLSAMPLES"]
all_figs = []
row_titles = []
all_legends = []
all_input_list = [sys.argv[1],] if len(sys.argv) > 1 else list(all_inputs)
for input_dataset_name  in tqdm(all_input_list):
    value = best_angles[input_dataset_name]
    print (input_dataset_name)
    fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
    mapping_legend = {}
    with open(fpath, 'rb') as handle:
        results_data = pickle.load(handle)
    if input_dataset_name == 'PPMI_FOR_ALIGNED_TIME_SERIES':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'Subtypes':
            num_subjects = 1
            selected_cats = ['PD_h', 'HC']
            mapping_legend = {'PD_h': 'High Parkinson Disease', 'HC': 'Healthy'}
            alpha = 1
        else:
            import sys; sys.exit()

    elif input_dataset_name == 'ADNI_FOR_ALIGNED_TIME_SERIES':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'Subtypes':
            num_subjects = 1
            selected_cats = ['Dementia', 'Control']
            mapping_legend = {'Control': 'Healthy', 'Dementia': 'Dementia'}
            alpha = 1
        else:
            import sys; sys.exit()

    elif input_dataset_name == 'AllN2_BioReactor_2D_3D_Report_protein':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'label':
            num_subjects = 1
            selected_cats = []
            alpha = 1
        else:
            import sys; sys.exit()

    elif input_dataset_name == 'ALVEOLAR_metacelltype' or input_dataset_name == 'NORM-ALVEOLAR_metacelltype':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'metacelltype':
            num_subjects = 0.7
            selected_cats = ['endothelial_cells', 'mesothelia_cells']
            alpha = 1
        elif colorname == 'cell.type':
            num_subjects = 0.7
            alpha = 1
            selected_cats = ['AT2 cells',  'Mesothelial cells', 'T-lymphocytes'] # 'B-lymphocytes',
        else:
            import sys; sys.exit()

        results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]


    elif input_dataset_name == 'COVID19Proteomics' or input_dataset_name == 'NORM-COVID19Proteomics':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'Acuity_max':
            mapping_legend = {1: 'Deceased', 2: 'Ventilated', 3: 'Hospitalized (O2)', 4: 'Hospitalized', 5: 'Recovered'}
            num_subjects = 1
            selected_cats = [1, 2, 3, 4, 5]
            alpha = 1
        else:
            import sys; sys.exit()

    elif input_dataset_name == 'MINMAX_MIMIC_ALLSAMPLES':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'y_true':
            num_subjects = 0.1
            selected_cats = []
            mapping_legend = {1: 'Readmission', 0: 'No readmission'}
            alpha = 1

        if colorname == 'MORTALITY_INHOSPITAL':
            num_subjects = 0.5
            colorname = 'MORTALITY_INHOSPITAL'
            mapping_legend = {1: 'Deceased', 0: 'Recovered'}
            selected_cats = [1]
            alpha = 1
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

        elif colorname == 'MORTALITY_INHOSPITAL_MALE':
            num_subjects = 0.8
            colorname = 'MORTALITY_INHOSPITAL'
            mapping_legend = {1: 'Deceased', 0: 'Recovered'}
            selected_cats = []
            alpha = 1
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER'] == 'M']

        elif colorname == 'LAST_CAREUNIT_MALE':
            colorname = 'LAST_CAREUNIT'
            num_subjects = 0.1
            selected_cats = ['MICU', 'CSRU']
            mapping_legend = {'MICU': 'Medical Intensive ICU (Male)', 'CSRU': 'Cardiac Surgery Recovery Unit (Male)'}
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER'] == 'M']
            # results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            alpha = 1

        elif colorname == 'LAST_CAREUNIT_FEMALE':
            colorname = sys.argv[2] if len(sys.argv) > 2 else 'LAST_CAREUNIT'
            # colorname = 'LAST_CAREUNIT'
            num_subjects = 0.1
            selected_cats = ['MICU', 'CSRU']
            mapping_legend = {'MICU': 'Medical Intensive ICU (Male)', 'CSRU': 'Cardiac Surgery Recovery Unit (Male)'}
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER'] == 'F']
            # results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            alpha = 1

        elif colorname == 'GENDER_LAST_CAREUNIT':
            num_subjects = 0.1
            results_data['complete_dataframe']['GENDER_LAST_CAREUNIT'] = results_data['complete_dataframe']['GENDER'] + '_' +  results_data['complete_dataframe']['LAST_CAREUNIT']
            selected_cats = ['M_MICU', 'M_CSRU', 'F_MICU', 'F_CSRU']
            mapping_legend = {}
            alpha = 1
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

        elif colorname == 'LOS-Class':
            num_subjects = 0.1
            results_data['complete_dataframe']['LOS-Class'] = results_data['complete_dataframe']['LOS'].map(lambda x: 'S' if x <= 7 else 'L')
            mapping_legend = {'L': 'Longer Stay (>7 days)', 'S': 'Shorter Stay (<7 days)'}
            selected_cats = ['L', 'S']
            alpha = 1
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

        elif colorname == 'GENDER_LOS-Class':
            num_subjects = 0.1
            results_data['complete_dataframe']['LOS-Class'] = results_data['complete_dataframe']['LOS'].map(lambda x: 'S' if x <= 7 else 'L')
            results_data['complete_dataframe']['GENDER_LOS-Class'] = results_data['complete_dataframe']['GENDER'] + '_' +  results_data['complete_dataframe']['LOS-Class']
            mapping_legend = {'L': 'Longer Stay (>7 days)', 'S': 'Shorter Stay (<7 days)'}
            mapping_legend = {}
            selected_cats = ['M_L', 'M_S', 'F_S', 'F_L']
            alpha = 1
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]

        else:
            import sys; sys.exit()

        # results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['GENDER']=='M']
        # results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe']['z']<=6]
        results_data['complete_dataframe'] = results_data['complete_dataframe'].drop_duplicates(subset=['z', 'subject_id', 'input_parameter_id'])

    elif input_dataset_name == 'IMAGENORM-PPMI-ADNI':
        colorname = sys.argv[2] if len(sys.argv) > 2 else color_name[input_dataset_name]
        if colorname == 'DIAGNOSIS':
            num_subjects = 0.4
            selected_cats = ['Dementia', 'Control']
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            alpha = 1
        elif colorname == 'GENDER':
            num_subjects = 0.4
            selected_cats = ['M', 'F']
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            alpha = 1
        elif colorname == 'GENDER_DIAGNOSIS':
            num_subjects = 0.4
            selected_cats = ['M_Dementia', 'F_Dementia', 'M_Control', 'F_Control']
            results_data['complete_dataframe']['GENDER_DIAGNOSIS'] = results_data['complete_dataframe']['GENDER'] + '_' +  results_data['complete_dataframe']['DIAGNOSIS']
            results_data['complete_dataframe'] = results_data['complete_dataframe'][results_data['complete_dataframe'][colorname].isin(selected_cats)]
            alpha = 1
        else:
            import sys; sys.exit()
    for eye_view in value[:2]:
        fig, (color_maps, nocolor_maps) = generate_dataset(results_data, input_dataset_name,  best_parameters[input_dataset_name], eye_view , num_subjects, selected_cats, colorname, scale=2, alpha=alpha, transparency_effect=True)
        all_figs.append(fig)
    all_legends.append(Image.open(get_legend_box(nocolor_maps, selected_cats, scale=4, mapping_legend=mapping_legend)))
    row_titles.append(name_mapping[input_dataset_name])

if not len(sys.argv) <= 1:
    os.makedirs(f"../final_images/MultiView_{input_dataset_name}", exist_ok=True)
    save_as_pdf_hbox(all_figs, row_titles, f'../final_images/MultiView_{input_dataset_name}/{sys.argv[2]}.pdf', ncols=3, scale=2, all_legends=all_legends)
else:
    save_as_pdf_hbox(all_figs, row_titles, f'../final_images/MultiView_ALL.pdf', ncols=3, scale=4, all_legends=all_legends)
