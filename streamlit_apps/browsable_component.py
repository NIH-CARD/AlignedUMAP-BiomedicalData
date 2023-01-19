import pickle
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import streamlit as st
from collections import defaultdict
import plotly.io as pio
import plotly.express as px
pio.templates.default = pio.templates["plotly_white"] # "plotly_white"
SPECS = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': {
        'dataset_name': "Parkinson's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['PD_h', 'PD_l', 'PD_m',  'HC'],
        'mapping_legend': {'PD_h': "High Parkinson's Disease", 'HC': 'Healthy Controls', 'PD_l': "Low PD", 'PD_m': "Medium PD"},
        'annotation_legend': {'PD_h': 'High PD', 'HC': 'Healthy Controls'},
        'time_mapping': ['Years from baseline', {'t0':'Y0', 't1':'Y1', 't2':'Y2', 't3':'Y3', 't4':'Y4', 't5':'Year5',
                                                 't00':'Y0', 't01':'Y1', 't02':'Y2', 't03':'Y3', 't04':'Y4', 't05':'Year5'}],
        'alpha': 1
    },
    'ADNI_FOR_ALIGNED_TIME_SERIES':  {
        'dataset_name': "Alzheimer's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['Dementia', 'Control', 'MCI'],
        'mapping_legend': {'Control': 'Healthy Controls', 'Dementia': 'Dementia', 'MCI': 'Mild Cognitive Impairment'},
        'annotation_legend': {'Dementia': 'Dementia', 'Control': 'Healthy Controls'},
        'time_mapping': ['Months from baseline', {'t0': 'M0', 't1': 'M6', 't2': 'M12', 't3': 'Month24',
                                                   'bl': 'M0', 'm06': 'M6', 'm12': 'M12', 'm24': 'Month24'}],
        'alpha': 1
    },
    'NORM-ALVEOLAR_metacelltype':  {
        'dataset_name': "whole lung scRNA-seq",
        'colorname': 'cell.type',
        'num_subjects': 1,
        'selected_cats': ['AT2 cells', 'Mesothelial cells', 'T-lymphocytes'],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Days after bleomycin injury',
                         {
                             't0':'d0', 't1':'d3', 't2':'d7', 't3':'d10', 't4':'d14', 't5':'d21', 't6': 'day28',
                            't00':'d0', 't01':'d3', 't02':'d3', 't03':'d7', 't04':'d10', 't05':'d14', 't06': 'day21'
                          }],
        'alpha': 1
    },
    'IMAGENORM-PPMI-ADNI':  {
        'dataset_name': "Brain T1-MRI imaging for AD and PD",
        'colorname': 'GENDER_DIAGNOSIS',
        'num_subjects': 1,
        'selected_cats': ['M_Dementia', 'F_Dementia', 'M_PD', 'F_PD'],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Age', { }],
        'alpha': 1
    },
    'MINMAX_MIMIC_ALLSAMPLES':  {
        'dataset_name': "MIMIC III EHR data",
        'colorname': 'LAST_CAREUNIT',
        'num_subjects': 0.3,
        'selected_cats': ['MICU', 'CSRU'],
        'mapping_legend': {'MICU': 'Medical Intensive ICU (Male)', 'CSRU': 'Cardiac Surgery Recovery Unit (Male)'},
        'annotation_legend': {'MICU': 'MICU (M)', 'CSRU': 'CSRU (M)'},
        'time_mapping': ['Hours after ICU admission',
                         {
                             't0':'H0', 't1':'H12', 't2':'H24', 't3':'H36', 't4':'H48', 't5':'H60', 't6': 'Hour72',
                            'H0000':'H0', 'H0001':'H12', 'H0002':'H24', 'H0003':'H36', 'H0004':'H48', 'H0005':'H60', 'H0006': 'Hour72'
                         }
                ],
        'alpha': 1
    },
    'NORM-COVID19Proteomics':  {
        'dataset_name': "Proteomics COVID19 data",
        'colorname': 'Acuity_max',
        'num_subjects': 1,
        'selected_cats': [], #['1','2','4','5', 1,2,4,5], # 'Severe', 'Non-severe'
        'mapping_legend': {1: 'Severe',2:'Severe',3:'Non-severe',4:'Non-severe',5:'Non-severe',
                                 '1': 'Severe','2':'Severe','3':'Non-severe','4':'Non-severe','5':'Non-severe',
                                },# 2: 'Severe', 3: 'Non-severe'},
        'annotation_legend': {}, # 2: 'Severe', 3: 'Non-severe'},
        'manual_label_mapping': {1: 'Severe',2:'Severe',3:'Non-severe',4:'Non-severe',5:'Non-severe',
                                 '1': 'Severe','2':'Severe','3':'Non-severe','4':'Non-severe','5':'Non-severe',
                                },
        'time_mapping': ['Days after COVID Infection',
                         {
                             't0':'d0', 't1':'d3', 't2':'day7',
                             't3':'d3', 't7':'day7',
                         }
                ],
        'alpha': 1
    },
    'AllN2_BioReactor_2D_3D_Report_protein':  {
        'dataset_name': "Proteomics Stem Cell data (different culture)",
        'colorname': 'label',
        'num_subjects': 1,
        'selected_cats': [],
        'mapping_legend': {},
        'annotation_legend': {},
        'time_mapping': ['Days',
                         {
                            't0':'d0', 't1':'d3', 't2':'d7', 't3':'d14', 't4':'d21', 't5':'day28', 't6': 'day28',
                            't00':'d0', 't01':'d3', 't02':'d7', 't03':'d14', 't04':'d21', 't05':'day28', 't06': 'day28'
                         }
                ],
        'alpha': 1
    },
}


def app(input_dataset_name, color_column_list, metadata_descriptions, combined_best_parameters, **kwargs):
    color_sequence = list(px.colors.qualitative.G10) * 3
    color_sequence_seq = px.colors.sequential.Viridis
    input_color_column = st.sidebar.selectbox('Select the color factor', color_column_list, index=0)
    st.sidebar.info("***{}***: {}".format(input_color_column, metadata_descriptions.get(input_color_column, 'Self Explanatory')))

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

    input_visualization_method = "umap_aligned"
    with open(f"results_data/{input_dataset_name}/{input_visualization_method}/generated_data/{input_dataset_name}_16.pickle",'rb') as handle:
        results_data = pickle.load(handle)

    data = results_data['complete_dataframe']
    time_sequence = results_data['age_list']
    list_parameters = results_data['complete_dataframe']['input_parameter_id'].unique()
    input_parameter_name = combined_best_parameters[input_dataset_name][1]
    markers_flag = False
    l = sns.color_palette("viridis", 91)
    mycolorscale = list(l.as_hex())
    def hex_to_rgba(h, alpha):
        return tuple([int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)] + [alpha])

    def hex_to_rgba_tuple(h, alpha):
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


    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def return_generated_figure(temp, colorname, color_map, title, camera_setup_ppmi=None, time_mapping={},
                                mode_str='lines', rever_color=False):
        temp = temp.dropna(subset=['x', 'y', 'z', colorname, 'subject_id'])
        fig = px.line_3d(temp, x="z", y="y", z="x", color=colorname, width=1200, height=900, line_group='subject_id',
                         markers=True if mode_str == "lines+markers" else False, color_discrete_map=color_map,
                         color_discrete_sequence=px.colors.qualitative.G10)
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
                xaxis_title=SPECS[input_dataset_name]['time_mapping'][0],# "time-axis",
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
                    ticktext=[SPECS[input_dataset_name]['time_mapping'][1].get(f"t{i}",f"t{i}")  for i in tickllist],
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
            # scene_camera=camera_setup_ppmi,  # dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
            autosize=True,  # True if camera_setup_ppmi else False,
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
        # fig.update_layout(showlegend=True)
        fig.update_xaxes(title_standoff=0)
        return fig


    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def generate_dataset(input_dataset_name, parameter_fix, anglelist, num_subjects, selected_cats, colorname, scale=1,
                         alpha=1, show_plot=True):
        selected_cats = SPECS[input_dataset_name]['selected_cats']
        fpath = f"results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
        with open(fpath, 'rb') as handle:
            results_data = pickle.load(handle)
        data = results_data['complete_dataframe']  # .copy()
        data = data[data['input_parameter_id'] == parameter_fix]
        if not len(selected_cats) == 0:
            data = data[data[input_color_column].isin(selected_cats)]
        data[input_color_column] = data[input_color_column].replace(SPECS[input_dataset_name]['mapping_legend'])
        all_subjects = list(data['subject_id'].unique())
        random.seed(42)
        random.shuffle(all_subjects)
        print('All subject', len(all_subjects), 'Selected_subjects', num_subjects)
        # st.write('All subject', len(all_subjects), 'Selected_subjects', num_subjects)


        if num_subjects < 1:
            X = data.drop_duplicates(subset=['subject_id', colorname])
            y = X[colorname].values
            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_subjects, random_state=0)
            train_index, test_index = list(sss.split(X.values, y))[0]
            select_subject = list(set(list(X.iloc[list(test_index)]['subject_id'])))[:300]
            time_mapping = {}
            data = data[data['subject_id'].isin(select_subject)]
            selected_data = data  # .copy()
            color_maps, nocolor_maps = generate_color_map(selected_data[colorname], alpha=alpha)
        else:
            if num_subjects > 1:
                select_subject = all_subjects[:num_subjects]
            else:
                select_subject = all_subjects
            data = data[data['subject_id'].isin(select_subject)]
            time_mapping = {}
            selected_data = data  # .copy()
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
        if show_plot:
            fig = return_generated_figure(selected_data, colorname, color_maps, '', camera_setup_ppmi=camera_setup_ppmi,
                                          time_mapping={}, mode_str='lines+markers', rever_color=False)
        else:
            fig = None
        cached_dict = {'f1': fig, 'color_maps': color_maps, 'nocolor_maps': nocolor_maps}
        return cached_dict


    selected_cats = []
    show_plot = False
    # if st.sidebar.checkbox("Show line plot"):
    #    show_plot = True
    num_subjects = st.sidebar.slider('Select the number of samples to show', min_value=0.2, max_value=1.0, value=0.2, step=0.2) if show_plot else 1.0
    alpha = st.sidebar.slider('Select the transparency', min_value=0.2, max_value=1.0, value=1.0, step=0.2) if show_plot else 1.0
    charts = generate_dataset(input_dataset_name, input_parameter_name, combined_best_parameters[input_dataset_name][0],
                              num_subjects, selected_cats, input_color_column, scale=4, alpha=alpha, show_plot=show_plot)
    input_parameter_name = input_dataset_name + "-" + input_parameter_name
    data = results_data['complete_dataframe']
    if not len(SPECS[input_dataset_name]['selected_cats']) == 0:
        data = data[data[input_color_column].isin(SPECS[input_dataset_name]['selected_cats'])]
    data[input_color_column] = data[input_color_column].replace(SPECS[input_dataset_name]['mapping_legend'])
    data[input_color_column] = data[input_color_column].replace(SPECS[input_dataset_name]['mapping_legend'])
    time_sequence = results_data['age_list']
    color_codes = data[input_color_column]
    results_data[input_parameter_name]['data'][input_color_column] = results_data[input_parameter_name]['data'][
        input_color_column].map(lambda x: f"{x}")
    x = results_data[input_parameter_name]['data'][input_color_column]
    if len(results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.unique()) > 10:
        color_sequence = color_sequence_seq

    y = results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.map(
        lambda x: color_sequence[x % len(color_sequence)])
    color_mapping = charts['color_maps']
    age_list = results_data['age_list']
    age_list_mapping = list(map(lambda x: time_axis_mapping.get(x, x), age_list))
    traces = []
    legend_track = {}
    enm = 1
    animation_data = defaultdict(list)

    selected_data = results_data[input_parameter_name]['data']  # .copy()
    if not len(SPECS[input_dataset_name]['selected_cats']) == 0:
        selected_data = selected_data[selected_data[input_color_column].isin(SPECS[input_dataset_name]['selected_cats'])]
    selected_data[input_color_column] = selected_data[input_color_column].replace(SPECS[input_dataset_name]['mapping_legend'])
    selected_data[input_color_column] = selected_data[input_color_column].map(lambda x: f"{x}")
    selected_data = selected_data[~(selected_data[input_color_column].str.contains('UNK'))]
    from utils.cylinder_plot import get_cylinder_plot

    trace_inputs = []

    st.write('### Trajectory Visualization')


    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def generate_plot(fig):
        # fig = go.Figure(data=traces)
        grid_color = 'white'
        fig.update_layout(
            width=800,
            height=800,
            scene=dict(
                aspectratio=dict(x=0.5, y=1.25, z=0.5),
                yaxis_title=SPECS[input_dataset_name]['time_mapping'][0],# "time-axis",
                xaxis_title="UMAP-X",
                zaxis_title="UMAP-Y",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(age_list_mapping))),
                    # ticktext=age_list_mapping,
                    ticktext=[SPECS[input_dataset_name]['time_mapping'][1].get(i, i)  for i in age_list_mapping],
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3,
                ),
                xaxis=dict(
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3,
                ),
                zaxis=dict(
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3,
                )
            ),
            scene_camera=dict(eye=dict(x=0.5, y=0.8, z=0.75)),
            autosize=True,
        )
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.8,
            xanchor="left",
            x=0.25,
        ), dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        return fig


    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def generate_plot_basic(fig):
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="left",
            x=0.25,
        ), dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
        # fig.update_layout(modebar_remove=['zoom', 'pan'])
        return fig


    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def generate_plot_gaussian(fig, dummy=None):
        grid_color = 'white'
        # fig.update_traces(showlegend=True)
        fig.update_layout(
            width=800,
            height=800,
            scene=dict(
                aspectratio=dict(y=0.5, x=1.25, z=0.5),
                # y - x
                xaxis_title=SPECS[input_dataset_name]['time_mapping'][0],# "time-axis",
                yaxis_title="UMAP-X",
                zaxis_title="UMAP-Y",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(age_list_mapping))),
                    # ticktext=age_list_mapping,
                    ticktext=[SPECS[input_dataset_name]['time_mapping'][1].get(i, i) for i in age_list_mapping],
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3
                ),
                zaxis=dict(
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3
                ),
                yaxis=dict(
                    backgroundcolor=grid_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3
                )
            ),
            autosize=True,
        )
        fig.layout.scene.camera.projection.type = "orthographic"
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=10))
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="left",
            x=0.25,
        ), dragmode=False)
        fig.layout.xaxis.fixedrange = False
        fig.layout.yaxis.fixedrange = False
        # fig.update_layout(modebar_remove=['zoom', 'pan'])
        return fig


    fig = get_cylinder_plot(selected_data, input_color_column, color_mapping_original=color_mapping, height=800, width=600)
    config = {
        'displayModeBar': True,
        'staticPlot': False,
        'scrollZoom': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
            'filename': 'umap3Dplot',
            'height': 1200,
            'width': 1200,
            'scale': 1
        },
        'frameMargins': {
            'max': 0
        }
    }

    if show_plot:
        cols = st.columns([0.5, 4, 0.5, 4])
        cols[1].plotly_chart(generate_plot_gaussian(fig, dummy=selected_data), config=config, use_container_width=True)
        cols[-1].plotly_chart(generate_plot_basic(charts['f1']), config=config, use_container_width=True)
    else:
        # config['scrollZoom'] = True
        cols = st.columns([1, 8, 1])
        cols[1].plotly_chart(generate_plot_gaussian(fig, dummy=selected_data), config=config, use_container_width=True)

    st.write("***Note***: The shown visualization plots are generated for optimal hyperparameters of Aligned-UMAP using manually inspection. Next section allows to explore the effect of hyperparameters on visualization space. The optimal hyperparameters are ***{}***".format(input_parameter_name.split('-')[1].replace(';', '; ')))
    st.markdown("""---""")
