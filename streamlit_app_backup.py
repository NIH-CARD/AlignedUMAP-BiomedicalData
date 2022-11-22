import plotly.express as px
import numpy as np
import pandas as pd
import json
import numpy as np
import platform
if '3.7' in platform.python_version():
    import pickle5  as pickle
else:
    import pickle

import streamlit as st
import plotly.graph_objects as go
import base64
from collections import defaultdict
import plotly.io as pio
import plotly.express as px

pio.templates.default = pio.templates["plotly_white"] # "plotly_white"

max_width = 3000
padding_top = 10
padding_right = 5
padding_left = 5
padding_bottom = 10
COLOR = 'black'
BACKGROUND_COLOR = 'white'
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


from pathlib import Path
import os

data_dir = Path("/Users/dadua2/projects")

color_sequence = list(px.colors.qualitative.G10)  * 3
color_sequence_seq = px.colors.sequential.Viridis # list(px.colors.sequential.viridis)
# st.write(color_sequence)
input_dataset_name = "PPMI_FOR_ALIGNED_TIME_SERIES"
input_visualization_method = "umap_aligned"
input_color_column = 'subtype_category'

dataset_name_mapping_all = {
    # "Stem Cell Datasets":   {
    #         "stemCellData": "Single Cell Proteomic Data",
    #         "stemCellData@General-Precursor": "Single Cell Proteomic Data (General-Precursor)",
    #         "stemCellData@Specific-Cholinergic": "Single Cell Proteomic Data (Specific-Cholinergic)",
    #         "stemCellData@Specific-Astrocyte": "Single Cell Proteomic Data (Specific-Astrocyte)",
    # },
    # "Longitudinal Proteomics Datasets":   {
    #     "AllN2_BioReactor_2D_3D_Report_protein": "AllN2_BioReactor_2D_3D_Report_protein",
    #     "AllN2_Proteomics_BioReactor_2D_3D_Report_protein": "AllN2_Proteomics_BioReactor_2D_3D_Report_protein",
    #     "BioReactor_2D_3D_Report_protein": "BioReactor_2D_3D_Report_protein",
    #         "BioReactor_3D_3D_Report_protein": "BioReactor_3D_3D_Report_protein",
    #         "ProteinCOVID19Proteomics": "ProteinCOVID19Proteomics",
    #         "COVID19Proteomics": "COVID19 olink",
    # },
    "All Datasets": {
        "COVID19Proteomics": "COVID19 olink",
        "ALVEOLAR_metacelltype": "ALVEOLAR_metacelltype"
    },
    "Subtyping Medical Datasets":   {
            "PPMI_FOR_ALIGNED_TIME_SERIES": "PPMI Clinical Measurements",
            "ADNI_FOR_ALIGNED_TIME_SERIES": "ADNI Clinical Measurements",
    },
    "Single Cell RNA": {
        "ALVEOLAR_metacelltype": "ALVEOLAR_metacelltype"
    },
    # "Electronic Health Records": {
    #         "TESTMIMIC": "MIMC dataset for 6000 subjects",
    # },

    # "MRI Imaging Datasets":   {
    #         "NEWAGECOHORT-PPMI-ADNI": "T1 MRI Features from ADNI/PPMI",
    # },
    # # "Audio Medical Datasets": {
    # #        "Heartbeat": "Heartbeat",
    # # },
    # # "EEG Medical Datasets": {
    # #        "MotorImagery": "Motor Imagery Movement (finger/tongue)",
    # #        "FingerMovements": "Finger Movements (left/right)",
    # # },
    # "ECG Medical Datasets": {
    #         "LONGECG": "Heart rate from ECG Data",
    #         "PTBECG": "Spectrogram from ECG",
    #         # "StandWalkJump": "StandWalkJump",
    # },
    # # "Human Activity Recognition Medical Datasets": {
    # #        "NATOPS": "NATOPS",
    # #        "Epilepsy": "Epilepsy Detection"
    # # }
}



combined_best_parameters = {
        'PPMI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25],
                                         'metric=euclidean;alignment_regularisation=0.003;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
        'ADNI_FOR_ALIGNED_TIME_SERIES': [[1.25, -1.25, -1.25],
                                         'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=10;min_dist=0.10;num_cores=16'],
        'ALVEOLAR_metacelltype': [[-1.25, -1.25, 1.25], 'metric=euclidean;alignment_regularisation=0.030;alignment_window_size=2;n_neighbors=05;min_dist=0.10;num_cores=16'],
        'COVID19Proteomics': [[-1.25, -1.25, 1.25], 'metric=cosine;alignment_regularisation=0.030;alignment_window_size=3;n_neighbors=05;min_dist=0.01;num_cores=16']
}

color_columns = {
        'PPMI_FOR_ALIGNED_TIME_SERIES': ['Subtypes'],
        'ADNI_FOR_ALIGNED_TIME_SERIES': ['Subtypes'],
        'ALVEOLAR_metacelltype': ['metacelltype'],
        'COVID19Proteomics': ['Acuity_max']
}

# all_available_datasets = ["NATOPS", "Heartbeat", "StandWalkJump", "ECG5000", "MIMIC_SYNTHETIC", "PPMI_FOR_ALIGNED_TIME_SERIES", "ADNI_MRI_CORTICAL_MEASURES", "ADNI_FOR_ALIGNED_TIME_SERIES", "SINGLE_CELL_RNA", "pca_slices_right_hemi_reg"]
# all_available_datasets = [  "stemCellData", "PPMI_FOR_ALIGNED_TIME_SERIES", "Heartbeat", "ADNI_FOR_ALIGNED_TIME_SERIES", "NATOPS"] #"ADNI_FOR_ALIGNED_TIME_SERIES",  "Heartbeat", "ECG5000",  "NATOPS", "MIMIC_SYNTHETIC"]


dataset_descriptions = {
   "PPMI_FOR_ALIGNED_TIME_SERIES": "The dataset contains multivariate time series data to monitor Parkinson's disease and is used for identification of subtypes. The data is obtained from Parkinson’s Progression Markers Initiative (PPMI) database.",
   "stemCellData": "The dataset presents the longitudinal changes in the expression of stem cell starting from iPSC cell.",
   "ADNI_FOR_ALIGNED_TIME_SERIES": "The dataset contains multivariate time series data to monitor Alzheimer's disease and is used for identification of subtypes. The data is obtained from Alzheimer's Disease Neuroimaging Initiative (ADNI) database.",
   "MIMIC_SYNTHETIC": "The dataset contains multi-variate electronic health records data to monitor patient's health. The data is obtained from Alzheimer's Disease Neuroimaging Initiative (ADNI) database.",
   "Heartbeat": "This dataset is publicly available at http://timeseriesclassification.com This dataset is derived from the PhysioNet/CinC Challenge 2016. Heart sound recordings were sourced from several contributors around the world, collected at either a clinical or nonclinical environment, from both healthy subjects and pathological patients. The heart sound recordings were collected from different locations on the body.", 
   "ECG5000": "This dataset is publicly available at http://timeseriesclassification.com The original dataset for ECG5000 is a 20-hour long ECG downloaded from Physionet. The name is BIDMC Congestive Heart Failure Database(chfdb) and it is record chf07.",  
   "NATOPS": "This dataset is publicly available at http://timeseriesclassification.com This data was originally part of a competition (Link Here) The data is generated by sensors on the hands, elbows, wrists and thumbs. The data are the x,y,z coordinates for each of the eight locations.",
   "Epilepsy": "This dataset is publicly available at http://timeseriesclassification.com The data was generated with healthy participants simulating the class activities of performed. Data was collected from 6 participants using a tri-axial accelerometer on the dominant wrist whilst conducting 4 different activities." 

}

metadata_descriptions = {
    "Subtypes": "Subtypes are subdivision of a heterogeneous disorders having similar characteristics such as disease progression, drug response and others. Heterogeneous disorders include Alzheimer's Disease, Parkinson's Disease and others.",
    "Default": "Self Explanatory"
}
time_axis_mapping = {
    "BL": 'Baseline',
    'V04': "Visit 4",
    'V06': "Visit 6",
    'V08': "Visit 8",
    'V10': "Visit 10",
    'V12': "Visit 12",
    "bl": 'Baseline',
    "m06": 'Month 6',
    "m12": 'Month 12',
    "m24": 'Month 24',

}
st.header("Aligned-UMAP for Longitudinal Biomedical Datasets")
st.write ("### Input Dataset")

cols = st.columns(3)
input_dataset_modality = cols[0].selectbox('Select the Dataset Modality', list(dataset_name_mapping_all.keys()))
cols[0].info("***{}***".format(input_dataset_modality))

dataset_name_mapping =  dataset_name_mapping_all[input_dataset_modality]
all_available_datasets = list(dataset_name_mapping.keys()) 
input_dataset_name = cols[1].selectbox('Select the dataset', list(map(lambda x: dataset_name_mapping[x], all_available_datasets)))
rmapping = {j:i for i, j in dataset_name_mapping.items()}
cols[1].info("***{}***: {}".format(input_dataset_name, dataset_descriptions.get(rmapping[input_dataset_name], rmapping[input_dataset_name])))

reverse_dataset_name_mapping = {j:i for i, j in dataset_name_mapping.items()}
input_dataset_name = reverse_dataset_name_mapping.get(input_dataset_name) 
all_available_visualization_methods = ['umap_aligned', 'multidr']
input_visualization_method = "umap_aligned"
# input_visualization_method = st.sidebar.selectbox('Select the visualization method', all_available_visualization_methods)


with open(data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/{input_visualization_method}/generated_data/{input_dataset_name}_16.pickle", 'rb') as handle:
    results_data = pickle.load(handle) 

data = results_data['complete_dataframe']
time_sequence = results_data['age_list'] 
color_column_list = color_columns[input_dataset_name] # [col for col in data.columns if not col in ['x', 'y', 'z', 'subject_id', 'feature_group', 'input_parameter_id', 'category']]
# st.write ("##### Color Coded Factor")
input_color_column = cols[2].selectbox('Select the color factor', color_column_list, index=0)
cols[2].info("***{}***: {}".format(input_color_column, metadata_descriptions.get(input_color_column, 'Self Explanatory')))
st.markdown("""---""")

if True:
    # list_parameters = [key.split('-')[-1] for key in results_data if not key in ['complete_dataframe', 'age_list']]
    list_parameters = results_data['complete_dataframe']['input_parameter_id'].unique()
    # st.write(list_parameters)
    # if len(list_parameters) == 1 and list_parameters[0] == 'best':
    #     no_parameteric_avail = True
    # else:
    #     no_parameteric_avail = False
    #
    # if no_parameteric_avail:
    #     input_parameter_name = combined_best_parameters[input_dataset_name][1]
    # else:
    #     L = defaultdict(list)
    #     for val in list_parameters:
    #        for item in val.split(';'):
    #            L[item.split('=')[0]].append(item.split('=')[1])
    #     # input_parameter_name = st.selectbox('Select the  parameter name', list_parameters, index=0) # list_parameters[0] #
        # input_parameter_name = list_parameters[0]

    input_parameter_name = combined_best_parameters[input_dataset_name][1]
    markers_flag = False
    if st.sidebar.checkbox("Show Markers"):
        markers_flag = True

    # input_parameter_name = f"{input_dataset_name}-{input_parameter_name}"
    # if len(color_column_list) == 0:
    #    input_color_column = "NONE"
    #    data['NONE'] = ['NO COLOR']*len(data)
    #    results_data[input_parameter_name]['data']['NONE'] = ['NO COLOR']*len(results_data[input_parameter_name]['data'])



    ###### UPDATE DATA


    from sklearn.model_selection import StratifiedShuffleSplit

    import pickle
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import random
    import plotly.express as px
    import seaborn as sns
    import plotly

    l = sns.color_palette("viridis", 91)
    mycolorscale = list(l.as_hex())
    from pathlib import Path

    import seaborn as sns


    def hex_to_rgba(h, alpha):
        '''
        converts color value in hex format to rgba format with alpha transparency
        '''
        return tuple([int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)] + [alpha])


    def hex_to_rgba_tuple(h, alpha):
        '''
        converts color value in hex format to rgba format with alpha transparency
        '''
        return tuple([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])


    def generate_color_map(category_list, alpha=1):
        category_list_unique = category_list.unique()
        if len(category_list_unique) <= 24:
            temp_mycolorscale = px.colors.qualitative.Dark24
            mycolorscale = []
            no_mycolorscale = []
            for enm in temp_mycolorscale:
                color = 'rgba' + str(hex_to_rgba(
                    h=enm,
                    alpha=alpha
                ))
                mycolorscale.append(color)
                nocolor = hex_to_rgba_tuple(
                    h=enm,
                    alpha=1
                )
                no_mycolorscale.append(nocolor)
            color_map = {}
            nocolor_map = {}
            for enm, i in enumerate(sorted(list(category_list_unique))):
                color_map[i] = mycolorscale[enm]
                nocolor_map[i] = no_mycolorscale[enm]
            return color_map, nocolor_map
        else:
            pass

#    @st.cache(allow_output_mutation=True)

    def return_generated_figure(temp, colorname, color_map, title, camera_setup_ppmi=None, time_mapping={},
                                mode_str='lines', rever_color=False):
        temp = temp.dropna(subset=['x', 'y', 'z', colorname, 'subject_id'])
        fig = px.line_3d(temp, x="z", y="y", z="x", color=colorname, width=1200, height=900, line_group='subject_id',
                         markers=True if mode_str == "lines+markers" else False, color_discrete_map=color_map, )
        # error_x = 'error_x', error_y = 'error_x', error_z = 'error_x')
        grid_color = 'white'
        bg_color = 'white'
        num = len(list(range(len(results_data['age_list']))))
        if num < 5:
            tickllist = np.arange(0, num, 1)
        else:
            tickllist = np.arange(0, num, num // 5)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                aspectratio=dict(x=1, y=0.5, z=0.5),
                xaxis_title="time-axis",
                zaxis_title="UMAP-X",
                yaxis_title="UMAP-Y",
                yaxis=dict(
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='black',
                    zerolinewidth=1,
                    showgrid=True,
                    zeroline=False,  # thick line at x=0
                    visible=True,  # numbers below
                    dtick=3,
                    ticks="outside",
                ),
                xaxis=dict(
                    tickvals=tickllist,
                    ticktext=[f"t{i}" for i in tickllist],
                    # [time_mapping.get( results_data['age_list'][xo], results_data['age_list'][xo]) for xo in tickllist], # list(map(lambda x: time_mapping.get(x, x), results_data['age_list'])),
                    # results_data['age_list'],
                    backgroundcolor=bg_color,
                    gridcolor='white',
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=1,
                    showgrid=True,
                    tickmode='array',
                    showspikes=False,
                    zeroline=False,
                    visible=True,
                    ticks="outside",
                ),
                zaxis=dict(
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='black',
                    zerolinewidth=1,
                    showgrid=True,
                    zeroline=False,  # thick line at x=0
                    visible=True,  # numbers below
                    showspikes=False,
                    dtick=3,
                    ticks="outside",
                )
            ),
            scene_camera=camera_setup_ppmi,  # dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
            autosize=False  # True if camera_setup_ppmi else False,
        )

        fig.update_traces(marker={'size': 1},
                          error_x=dict(
                              # type='constant',
                              # color='purple',
                              thickness=5,
                              width=3,
                          ),
                          error_y=dict(
                              # type='constant',
                              # color='purple',
                              thickness=5,
                              width=3,
                          ),
                          error_z=dict(
                              # type='constant',
                              # color='purple',
                              thickness=5,
                              width=3,
                          ),
                          )
        fig.update_layout(
            font=dict(
                size=12,
                color="black"
            )
        )
        fig.update_layout(showlegend=True)

        fig.update_xaxes(
            title_standoff=0
        )
        return fig



    @st.cache(hash_funcs={dict: lambda _: None})
    def generate_dataset(input_dataset_name, parameter_fix, anglelist, num_subjects, selected_cats, colorname, scale=1,
                         alpha=1):
        fpath = data_dir / f"projects.link/baseTimeVaryingAlignedUMAP/results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
        with open(fpath, 'rb') as handle:
            results_data = pickle.load(handle)

        data = results_data['complete_dataframe'].copy()
        # list_parameters = list(data['input_parameter_id'].unique())
        data = data[data['input_parameter_id'] == parameter_fix]

        all_subjects = list(data['subject_id'].unique())
        random.seed(42)
        random.shuffle(all_subjects)
        print('All subject', len(all_subjects), 'Selected_subjects', num_subjects)
        if num_subjects < 1:
            X = data.drop_duplicates(subset=['subject_id', colorname])
            y = X[colorname].values
            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_subjects, random_state=0)
            train_index, test_index = list(sss.split(X.values, y))[0]
            select_subject = list(set(list(X.iloc[list(test_index)]['subject_id'])))
            time_mapping = {}
            data = data[data['subject_id'].isin(select_subject)]
            selected_data = data.copy()
            color_maps, nocolor_maps = generate_color_map(selected_data[colorname], alpha=alpha)
        else:
            if num_subjects > 1:
                select_subject = all_subjects[:num_subjects]
            else:
                select_subject = all_subjects
            data = data[data['subject_id'].isin(select_subject)]
            time_mapping = {}
            selected_data = data.copy()
            color_maps, nocolor_maps = generate_color_map(selected_data[colorname], alpha=alpha)
        if len(selected_cats) == 0:
            pass
        else:
            selected_data = selected_data[
                selected_data[colorname].isin(selected_cats)]  # ['macrophages', 'alv_epithelium', 'T_cells']
        camera_setup_ppmi = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=anglelist[0], y=anglelist[1],
                     z=anglelist[2])
        )
        fig = return_generated_figure(selected_data, colorname, color_maps, '', camera_setup_ppmi=camera_setup_ppmi,
                                         time_mapping={}, mode_str='lines+markers', rever_color=False)
        cached_dict = {'f1': fig}
        return cached_dict

       #  return st.plotly_chart(charts['f1']), (color_maps, nocolor_maps)

    st.write(input_dataset_name)
    st.write(input_color_column)
    st.write(input_parameter_name)
    selected_cats = []
    num_subjects = st.sidebar.slider('Select the number of samples to show', min_value=0.2, max_value=1.0, value=0.2, step=0.2)
    alpha = st.sidebar.slider('Select the transparency', min_value=0.2, max_value=1.0, value=1.0,step=0.2)
    charts = generate_dataset(input_dataset_name, input_parameter_name, combined_best_parameters[input_dataset_name][0], num_subjects, selected_cats, input_color_column, scale=4, alpha=alpha)
    st.write(charts['f1'])
