from utils.cylinder_plot import get_cylinder_plot
import pickle
import streamlit as st
from collections import defaultdict
import plotly.io as pio
pio.templates.default = pio.templates["plotly_white"] # "plotly_white"
SPECS = {
    'PPMI_FOR_ALIGNED_TIME_SERIES': {
        'dataset_name': "Parkinson's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['PD_h', 'HC'],
        'mapping_legend': {'PD_h': "High Parkinson's Disease", 'HC': 'Healthy Controls'},
        'annotation_legend': {'PD_h': 'High PD', 'HC': 'Healthy Controls'},
        'time_mapping': ['Years from baseline', {'t0':'Y0', 't1':'Y1', 't2':'Y2', 't3':'Y3', 't4':'Y4', 't5':'Year5',
                                                 't00':'Y0', 't01':'Y1', 't02':'Y2', 't03':'Y3', 't04':'Y4', 't05':'Year5'}],
        'alpha': 1
    },
    'ADNI_FOR_ALIGNED_TIME_SERIES':  {
        'dataset_name': "Alzheimer's Disease Progression",
        'colorname': 'Subtypes',
        'num_subjects': 1,
        'selected_cats': ['Dementia', 'Control'],
        'mapping_legend': {'Control': 'Healthy Controls', 'Dementia': 'Dementia'},
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
        'annotation_legend': {},
        'selected_cats': [],
        'mapping_legend': {1: 'Severe',2:'Severe',3:'Non-severe',4:'Non-severe',5:'Non-severe',
                                 '1': 'Severe','2':'Severe','3':'Non-severe','4':'Non-severe','5':'Non-severe',
                                },
        'manual_label_mapping': {'1': 'Severe',2:'Severe',3:'Non-severe',4:'Non-severe',5:'Non-severe'},
        'time_mapping': ['Days after COVID Infection',
                         {
                             't0':'d0', 't1':'d3', 't2':'day7',
                             't0':'d0', 't3':'d3', 't7':'day7',
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
    color_mapping = {}
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

    input_color_column = st.sidebar.selectbox('Select the color factor', color_column_list, index=0)
    st.sidebar.info("***{}***: {}".format(input_color_column, metadata_descriptions.get(input_color_column, 'Self Explanatory')))

    fpath = f"results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
    with open(fpath, 'rb') as handle:
        results_data = pickle.load(handle)
    age_list = results_data['age_list']
    age_list_mapping = list(map(lambda x: time_axis_mapping.get(x, x), age_list))
    list_parameters = results_data['complete_dataframe']['input_parameter_id'].unique()
    if len(list_parameters) <= 1:
        st.stop()

    cols = st.columns([4,1,4])
    cols[0].write("### Parameter Set-1")
    cols[2].write("### Parameter Set-2")
    for co in [0,2]:
        input_parameter_name_list = []
        plot_col = cols[co].empty()
        L = defaultdict(list)
        for val in list_parameters:
            for item in val.split(';'):
                L[item.split('=')[0]].append(item.split('=')[1])
        for item, val in dict(L).items():
            if len(sorted(list(set(list(map(str, val)))))) == 1:
                input_val = list(set(list(map(str, val))))[0]
            else:
                input_val = cols[co].select_slider(f"Select a value of {item}{' '*co}", sorted(list(set(list(map(str, val))))))
            input_parameter_name_list.append(f"{item}={'{}'.format(input_val)}")
        input_parameter_name = ';'.join(input_parameter_name_list)  # + ";num_cores=16"
        input_parameter_name = f"{input_dataset_name}-{input_parameter_name}"
        selected_data = results_data[input_parameter_name]['data']  # .copy()
        selected_data[input_color_column] = selected_data[input_color_column].map(lambda x: f"{x}")
        selected_data = selected_data[~(selected_data[input_color_column].str.contains('UNK'))]
        if not len(SPECS[input_dataset_name]['selected_cats']) == 0:
            selected_data = selected_data[selected_data[input_color_column].isin(SPECS[input_dataset_name]['selected_cats'])]

        selected_data[input_color_column] = selected_data[input_color_column].replace(SPECS[input_dataset_name]['mapping_legend'])
        fig = get_cylinder_plot(selected_data, input_color_column, color_mapping_original=color_mapping)

        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
        def generate_plot_gaussian(fig, dummy=None):
            grid_color = 'white'
            # fig.update_traces(showlegend=False)
            fig.update_layout(
                width=800,
                height=800,
                scene=dict(
                    aspectratio=dict(y=0.5, x=1.25, z=0.5),
                    # y - x
                    xaxis_title=SPECS[input_dataset_name]['time_mapping'][0], # "time-axis",
                    yaxis_title="UMAP-X",
                    zaxis_title="UMAP-Y",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(age_list_mapping))),
                        ticktext=[SPECS[input_dataset_name]['time_mapping'][1].get(i, i)  for i in age_list_mapping],# age_list_mapping,
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
            fig.layout.xaxis.fixedrange = True
            fig.layout.yaxis.fixedrange = True
            return fig



        fig = generate_plot_gaussian(fig)
        fig.update_layout(width=800, height=800)
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
                'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
            },
            'frameMargins': {
                'max': 0
            }
        }
        plot_col.plotly_chart(fig, config=config, use_container_width=True)
