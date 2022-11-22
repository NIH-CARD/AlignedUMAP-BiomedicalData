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
    "Longitudinal Proteomics Datasets":   {
        "AllN2_BioReactor_2D_3D_Report_protein": "AllN2_BioReactor_2D_3D_Report_protein",
        "AllN2_Proteomics_BioReactor_2D_3D_Report_protein": "AllN2_Proteomics_BioReactor_2D_3D_Report_protein",
        "BioReactor_2D_3D_Report_protein": "BioReactor_2D_3D_Report_protein",
            "BioReactor_3D_3D_Report_protein": "BioReactor_3D_3D_Report_protein",
            "ProteinCOVID19Proteomics": "ProteinCOVID19Proteomics",
            "COVID19Proteomics": "COVID19 olink",
    },
    "Subtyping Medical Datasets":   {
            "PPMI_FOR_ALIGNED_TIME_SERIES": "PPMI Clinical Measurements",
            "ADNI_FOR_ALIGNED_TIME_SERIES": "ADNI Clinical Measurements",
    },
    "Electronic Health Records": {
            "TESTMIMIC": "MIMC dataset for 6000 subjects",
    },

    "MRI Imaging Datasets":   {
            "NEWAGECOHORT-PPMI-ADNI": "T1 MRI Features from ADNI/PPMI",
    },
    # "Audio Medical Datasets": {
    #        "Heartbeat": "Heartbeat",
    # },
    # "EEG Medical Datasets": {
    #        "MotorImagery": "Motor Imagery Movement (finger/tongue)",
    #        "FingerMovements": "Finger Movements (left/right)",
    # },
    "ECG Medical Datasets": {
            "LONGECG": "Heart rate from ECG Data",
            "PTBECG": "Spectrogram from ECG",
            # "StandWalkJump": "StandWalkJump",
    },
    # "Human Activity Recognition Medical Datasets": {
    #        "NATOPS": "NATOPS",
    #        "Epilepsy": "Epilepsy Detection"
    # }
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


with open(f"results_data/{input_dataset_name}/{input_visualization_method}/generated_data/{input_dataset_name}_16.pickle", 'rb') as handle:
    results_data = pickle.load(handle) 

data = results_data['complete_dataframe']
time_sequence = results_data['age_list'] 
color_column_list = [col for col in data.columns if not col in ['x', 'y', 'z', 'subject_id', 'feature_group', 'input_parameter_id', 'category']]
# st.write ("##### Color Coded Factor")
input_color_column = cols[2].selectbox('Select the color factor', color_column_list, index=0)
cols[2].info("***{}***: {}".format(input_color_column, metadata_descriptions.get(input_color_column, 'Self Explanatory')))
st.markdown("""---""")

if input_visualization_method == 'multidr':
    data_Z_n_dt = data[data['z']=='Z_n_dt']
    st.write('### Clustering for similar trajectory')
    fig = px.scatter(data_frame=data_Z_n_dt, x='x', y='y', color=input_color_column, color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_layout(
        width=800,
        height=600,
        plot_bgcolor='white',
    )
    st.plotly_chart(fig)
elif input_visualization_method == 'umap_aligned':
    list_parameters = [key.split('-')[-1] for key in results_data if not key in ['complete_dataframe', 'age_list']]
    # st.write(list_parameters)
    if len(list_parameters) == 1 and list_parameters[0] == 'best':
        no_parameteric_avail = True
    else:
        no_parameteric_avail = False
    
    if no_parameteric_avail:
        input_parameter_name = "best"
    else:
        L = defaultdict(list)
        for val in list_parameters:
           for item in val.split(';'):
               L[item.split('=')[0]].append(item.split('=')[1])
        # input_parameter_name = st.selectbox('Select the  parameter name', list_parameters, index=0) # list_parameters[0] # 
        input_parameter_name = list_parameters[0]


    value_type = st.sidebar.radio("Select the values type:", ['raw','rolling mean']) #  'rolling variance',
    markers_flag = False
    if st.sidebar.checkbox("Show Markers"):
        markers_flag = True

    input_parameter_name = f"{input_dataset_name}-{input_parameter_name}"
    if len(color_column_list) == 0:
        input_color_column = "NONE"
        data['NONE'] = ['NO COLOR']*len(data)
        results_data[input_parameter_name]['data']['NONE'] = ['NO COLOR']*len(results_data[input_parameter_name]['data']) 
    
    trace_inputs = results_data[input_parameter_name]['trace_inputs']
    color_codes = data[input_color_column]
    for enm, rep in enumerate(results_data[input_parameter_name]['data'].subject_id.unique()):
        color_index = color_codes.iloc[enm] 
        for enm_iter in range(len(trace_inputs[enm])):
            bsize = trace_inputs[enm][enm_iter][0].shape[0]
            # print (trace_inputs[enm][enm_iter])
            # trace_inputs[enm][enm_iter] = list(pd.Series(trace_inputs[enm][enm_iter]).rolling(window=5, min_periods=1).std())
            trace_inputs[enm][enm_iter] = list(trace_inputs[enm][enm_iter])
            if value_type == 'rolling variance':
                trace_inputs[enm][enm_iter][1] = list(pd.Series(trace_inputs[enm][enm_iter][1]).rolling(window=5, min_periods=1).std())
                trace_inputs[enm][enm_iter][2] = list(pd.Series(trace_inputs[enm][enm_iter][2]).rolling(window=5, min_periods=1).std())
            elif value_type == 'rolling mean':
                trace_inputs[enm][enm_iter][1] = list(
                    pd.Series(trace_inputs[enm][enm_iter][1]).rolling(window=5, min_periods=1).mean())
                trace_inputs[enm][enm_iter][2] = list(
                    pd.Series(trace_inputs[enm][enm_iter][2]).rolling(window=5, min_periods=1).mean())

            trace_inputs[enm][enm_iter].append(np.array([color_index]*bsize))
            trace_inputs[enm][enm_iter].append(np.array([rep]*bsize))
    
    results_data[input_parameter_name]['data'][input_color_column] = results_data[input_parameter_name]['data'][input_color_column].map(lambda x: f"{x}")
    x = results_data[input_parameter_name]['data'][input_color_column]
    # st.write(results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes)
    if len(results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.unique()) > 10:
        color_sequence = color_sequence_seq
    y = results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.map(lambda x: color_sequence[x%len(color_sequence)])
    color_mapping = dict(zip(list(x), list(y)))

    age_list = results_data['age_list']
    age_list_mapping = list(map(lambda x: time_axis_mapping.get(x, x), age_list))
    if markers_flag:
        mode_str = "lines+markers"
    else:
        mode_str = "lines"
    traces = []
    legend_track = {}     
    enm = 1
    animation_data = defaultdict(list)
    import random
    random.seed(42)
    random.shuffle(trace_inputs)
    samples_to_show = st.sidebar.slider('Select the number of samples to show', min_value=1, max_value=len(trace_inputs), value=200)
    for enm, trace_input in enumerate(trace_inputs[:samples_to_show]):
        for z, x, y, _, c, text in trace_input:
            if 'SMAD_2' in text:
                continue
            # c = list(map(lambda zs: f"{zs}", c))
            animation_data['subject_id'].extend([f"{enm}"]*len(x))
            animation_data['x'].extend(x)
            animation_data['z'].extend(z)
            animation_data['y'].extend(y)
            animation_data['c'].extend(c)
            animation_data['text'].extend(text)
            # st.write(z,x,y,c,text, color_mapping)# , color_mapping["0"])
            trace = go.Scatter3d(
                x=x, y=z, z=y,
                mode=mode_str,
                hovertext=text,
                hoverinfo="text",
                name = str(c[0]),
                legendgroup=str(c[0]),
                line=dict(
                    color=list(map(lambda pol: color_mapping[str(pol)], c)),
                    width=2 if mode_str == "lines" else 1,
                ),
                marker=None, # dict(color=color_mapping[str(c[0])], size=4),
                opacity=0.7,
                showlegend=True if legend_track.get(str(c[0]), None) is None else False
            )
            traces.append(trace)
            legend_track[str(c[0])] = 1
        enm += 1
    animation_data  = pd.DataFrame(animation_data)
    # st.write(animation_data.head())

    camera_setup_ppmi = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0.5),
        eye=dict(x=0.25, y=0.3, z=1.6)
    )
    import pickle
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import random
    import plotly.express as px

    @st.cache(allow_output_mutation=True)
    def return_generated_figure(temp, colorname, title, camera_setup_ppmi=None, time_mapping={}, highlight_category=[]):
        if len(highlight_category) == 0:
            pass
        else:
            # animation_data = animation_data[animation_data['c'] == highlight_category]
            temp['c'] = temp['text'].map(lambda x: 1 if x in highlight_category else 0)
        temp = temp.dropna()
        # st.write(len(temp['c'].unique()))
        # if type(temp.iloc[0]['c']) is str or type(temp.iloc[0]['c']) is np.str_:
        #    if len(temp['c'].unique()) > 50:
        #         temp['c'] = ['C0'] * len(temp)
        if len(temp['c'].unique()) < 20:# and len(temp['c'].unique()) > 1:
            mycolorscale = px.colors.qualitative.Dark24
            if len(results_data['age_list']) > 80:
                mycolorscale = ["rgba(0,255,0,0.3)", "rgba(255,0,0,0.3)"]
            if not len(highlight_category) == 0:
                mycolorscale = ["rgba(0,255,0,0.03)", "rgba(0,0,0,1)", "rgba(255,0,0,1)"]

            makelegend = True
            pass
        else:
            # if len(temp['c'].unique()) == 1:
            #    subject_id_map = dict(zip(sorted(list(temp['subject_id'].unique())), list(range(len(temp['subject_id'].unique())))))
            #    temp['c'] = temp['subject_id'].map(lambda x: subject_id_map[x])
            #    pass
            # print (temp.iloc[0]['c'], type(temp.iloc[0]['c']), type(temp.iloc[0]['c']) is np.str_)
            if type(temp.iloc[0]['c']) is str or type(temp.iloc[0]['c']) is np.str_:
                temp['c'] = temp['c'].astype('category').cat.codes.map(int)

            temp['c'] = temp['c'].map(float).round(2).astype(int)  # .round(2)
            max_95 = np.percentile(temp['c'], 95)
            min_05 = np.percentile(temp['c'], 5)
            temp['c'] = temp['c'].map(lambda x: min(x, max_95))
            temp['c'] = temp['c'].map(lambda x: max(x, min_05))
            # data['period_length'] = data['period_length'].astype(int)
            temp['c'] = temp['c'].astype(int)# .round(2)
            import seaborn as sns
            l = sns.color_palette("YlOrBr", len(temp['c'].unique()))
            mycolorscale = list(l.as_hex())
            # mycolorscale = plotly.express.colors.n_colors('rgb(255, 0, 255)', 'rgb(0,255,0)', len(temp['c'].unique()), colortype='rgb')
            # mycolorscale = plotly.express.colors.n_colors('rgb(0, 0, 255)', 'rgb(0,255,0)', len(temp['c'].unique()), colortype='rgb')
            # mycolorscale  = colors=set_palette("Reds", 24)
            # mycolorscale = plotly.express.colors.n_colors('rgb(0, 0, 255)', 'rgb(255, 0, 0)', len(temp['c'].unique()), colortype='rgb')
            # mycolorscale = plotly.express.colors.n_colors(plotly.colors.hex_to_rgb(px.colors.sequential.Plasma[0]), plotly.colors.hex_to_rgb(px.colors.sequential.Plasma[-1]), len(temp['c'].unique()), colortype='rgb')
            temp = temp.sort_values(by='c')
            # temp['c'] = temp['c'].astype(str)
            makelegend = False
        fig = px.line_3d(temp, x="z", y="y", z="x", color=colorname, width=1200, height=1000, line_group='subject_id', hover_data=['text'],
                         markers=True if mode_str == "lines+markers" else False, color_discrete_sequence=mycolorscale)

        if not makelegend:
            colorbar_trace = go.Scatter(x=[None],
                                        y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale=mycolorscale,
                                            showscale=True,
                                            cmin=temp['c'].min(),
                                            cmax=temp['c'].max(),
                                            colorbar=dict(thickness=10, title=input_color_column, tickvals=[temp['c'].min(), temp['c'].max()], ticktext=[f"Low-{temp['c'].min()}", f"High-{temp['c'].max()}"]),
                                            # colorbar=dict(thickness=10, tickvals=[-5, 5], ticktext=['Low', 'High']),
                                        ),
                                        hoverinfo='none',
                                        # bg_color='white'
                                        )
            # colorbar_trace = px.colors.make_colorscale([[0, 'rgb(255,0,0)'], [1, 'rgb(0,0,255)']])
            # colorbar_trace.update_layout(plot_bgcolor='rgb(12,163,135)',)
            # fig.update_traces(colorscale="RdBu")
            # fig.update_traces(marker=dict(showscale=True, reversescale=True, cmin=6, size=20))

            fig_colorbar = go.Figure(data=[colorbar_trace], layout=go.Layout(plot_bgcolor='white', width=100, height=500, margin=dict(l=100, r=250, b=0, t=0)))
            fig_colorbar.update_xaxes(title='', visible=False, showticklabels=False)
            # Set the visibility OFF
            fig_colorbar.update_yaxes(title='', visible=False, showticklabels=False)



        else:
            fig_colorbar = None
            pass
        name = 'eye = (x:0., y:0, z:0), point along '
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0.1, y=0, z=0.5),
            eye=dict(x=0.5, y=0.25, z=1.5)
        )

        grid_color = 'lightgrey'
        bg_color = 'white'
        num = len(list(range(len(results_data['age_list']))))
        if num < 10:
            tickllist = np.arange(0, num, 1)
        else:
            tickllist = np.arange(0, num, num//10)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                aspectratio=dict(x=1, y=0.5, z=0.5),
                xaxis_title="time-axis",
                zaxis_title="UMAP-X",
                yaxis_title="UMAP-Y",
                yaxis=dict(
                    tickmode='array',
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3
                ),
                xaxis=dict(
                    tickvals=tickllist,
                    ticktext=[time_mapping.get( results_data['age_list'][xo], results_data['age_list'][xo]) for xo in tickllist], # list(map(lambda x: time_mapping.get(x, x), results_data['age_list'])),
                    # results_data['age_list'],
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3
                ),
                zaxis=dict(
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    zerolinecolor='white',
                    zerolinewidth=3,
                )
            ),
            # scene_camera=camera_setup_ppmi,  # dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
            autosize=True,
        )

        fig.update_traces(marker={'size': 3})
        fig.update_layout(legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=0.8,
                                      xanchor="right",
                                      x=0.8,
                                      title=input_color_column, ), showlegend=makelegend)

        fig.update_layout(
            font=dict(
                # family="Courier New, monospace",
                size=14,
                color="black"
            )
        )
        fig.update_xaxes(
            title_standoff=50
        )

        return fig, fig_colorbar


    # animation_data['z'] = animation_data['z'].map(lambda x: round(x, 2))
    # animation_data = animation_data.replace({'Dimentia': 'Dementia'})
    
    if True:
        st.write('### Visualization')
        st.write('##### Trajectory')
        @st.cache
        def generate_plot():
            fig = go.Figure(data=traces)
            grid_color = 'white'
            fig.update_layout(
            width=1000,
            height=600,
            scene=dict(
            aspectratio = dict( x=0.5, y=1.25, z=0.5 ),
            yaxis_title="time-axis",
            xaxis_title="UMAP-X",
            zaxis_title="UMAP-Y",
            yaxis = dict(
                tickmode = 'array',
                tickvals = list(range(len(age_list_mapping))),
                ticktext =  age_list_mapping,
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            ),
            xaxis = dict(
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            ),
            zaxis= dict(
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            )
            ),
            scene_camera=dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
            autosize=False,
            )
            fig.update_layout(
                font=dict(
                    # family="Courier New, monospace",
                    size=14,
                    color="black"
                )
            )
            return fig



        # st.write(fig)
        st.info("Rendering trajectory plot will take few minutes....")
        # try:
        #     animation_data['c'] = animation_data['c'].astype(float)
        # except:
        #     pass
        col1, col2 = st.columns(2)
        if col1.checkbox("Highlight"):
            highlight_category = col2.multiselect('Select category to highlight', list(set(animation_data['text'].tolist())), default=[])
        else:
            highlight_category = []
            if col2.checkbox("Show single plane"):
                plane_select = col2.selectbox('Select category to highlight',
                                                      list(set(animation_data['z'].tolist())))
                animation_data = animation_data[animation_data['z'].map(int)==int(plane_select)]
            else:
                highlight_category = []
        # if col2.button('Highlight Selected Protein'):
        fig, color_bar = return_generated_figure(animation_data, 'c', title='', highlight_category=highlight_category)
        # else:
        #     fig, color_bar = return_generated_figure(animation_data, 'c', title='', highlight_category= ['Select',])
        if color_bar:
            col1, col2 = st.columns((4, 1))
            col1.plotly_chart(fig)
            col2.plotly_chart(color_bar)
        else:
            st.plotly_chart(fig)

        # st.write('##### Change in lower dimensional space with time axis')
        # fig = px.scatter(animation_data, x="x", y="y", animation_frame="z", color="c", hover_name="text", color_discrete_sequence=px.colors.qualitative.G10)
        # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
        # fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
        # fig.update_layout(
        #     width=800,
        #     height=600,
        #     plot_bgcolor='white',
        # )
        # st.plotly_chart(fig)
    st.write ("***Note***: The shown visualization plots are generated for optimal hyperparameters of Aligned-UMAP using manually inspection. Next section allows to explore the effect of hyperparameters on visualization space. The optimal hyperparameters are ***{}***".format(input_parameter_name.split('-')[1].replace(';', '; ')))

    st.markdown("""---""")
    if no_parameteric_avail:
        st.stop()
    st.write('### Parameteric Effect')
    st.write("***Note:*** Select parameters from sidebar, then click Regenerate Plot.")
    st.sidebar.write("### Select Parameters")
    input_parameter_name_list = []
    flag = 0
    for item, val in dict(L).items():
        # st.write(val,  sorted(list(set(list(map(float, val))))))
        if len(val) == 1:
            continue
        else:
            flag = 1
        if item == 'num_cores':
            input_val = val[0]
        elif item == 'metric':
            input_val = st.sidebar.selectbox(item, list(set(val)))
        else:
            input_val = st.sidebar.select_slider( f'Select a value of {item}', options=sorted(list(set(val))), value=sorted(list(set(val)))[0],  )
        # input_parameter_name_list.append(f"{item}={'{:06.2f}'.format(input_val)}")
        input_parameter_name_list.append(f"{item}={'{}'.format(input_val)}")
    # st.write(input_parameter_name_list)
    input_parameter_name = ';'.join(input_parameter_name_list)
    # color = st.select_slider( 'Select a color of the rainbow', options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    # st.write('Selected parameter', input_parameter_name)
    if flag==0:
        st.sidebar.write("### No Parameters Available. Select other datasets.")
        st.stop()
    show_3d = True # st.checkbox('Show 3D Trajectory')
    if st.button("Regenerate Plot"):
        pass
    else:
        st.stop()

    input_parameter_name = f"{input_dataset_name}-{input_parameter_name}"
    if len(color_column_list) == 0:
        input_color_column = "NONE"
        data['NONE'] = ['NO COLOR']*len(data)
        results_data[input_parameter_name]['data']['NONE'] = ['NO COLOR']*len(results_data[input_parameter_name]['data']) 

    trace_inputs = results_data[input_parameter_name]['trace_inputs']
    color_codes = data[input_color_column]
    for enm, rep in enumerate(results_data[input_parameter_name]['data'].subject_id.unique()):
        color_index = color_codes.iloc[enm] 
        for enm_iter in range(len(trace_inputs[enm])):
            bsize = trace_inputs[enm][enm_iter][0].shape[0]
            trace_inputs[enm][enm_iter] = list(trace_inputs[enm][enm_iter])
            if value_type == 'rolling variance':
                trace_inputs[enm][enm_iter][1] = list(pd.Series(trace_inputs[enm][enm_iter][1]).rolling(window=5, min_periods=1).var())
                trace_inputs[enm][enm_iter][2] = list(pd.Series(trace_inputs[enm][enm_iter][2]).rolling(window=5, min_periods=1).var())
            elif value_type == 'rolling mean':
                trace_inputs[enm][enm_iter][1] = list(
                    pd.Series(trace_inputs[enm][enm_iter][1]).rolling(window=5, min_periods=1).mean())
                trace_inputs[enm][enm_iter][2] = list(
                    pd.Series(trace_inputs[enm][enm_iter][2]).rolling(window=5, min_periods=1).mean())

            trace_inputs[enm][enm_iter].append(np.array([color_index]*bsize))
            trace_inputs[enm][enm_iter].append(np.array([rep]*bsize))







    ## ---------------
    results_data[input_parameter_name]['data'][input_color_column] = results_data[input_parameter_name]['data'][
        input_color_column].map(lambda x: f"{x}")
    x = results_data[input_parameter_name]['data'][input_color_column]
    # st.write(results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes)
    if len(results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.unique()) > 10:
        color_sequence = color_sequence_seq
    y = results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.map(
        lambda x: color_sequence[x % len(color_sequence)])
    color_mapping = dict(zip(list(x), list(y)))

    age_list = results_data['age_list']
    age_list_mapping = list(map(lambda x: time_axis_mapping.get(x, x), age_list))

    traces = []
    legend_track = {}
    enm = 1
    animation_data = defaultdict(list)
    import random
    if markers_flag:
        mode_str = "lines+markers"
    else:
        mode_str = "lines"
    random.seed(42)
    random.shuffle(trace_inputs)
    for enm, trace_input in enumerate(trace_inputs[:samples_to_show]):
        for z, x, y, _, c, text in trace_input:
            if 'SMAD_2' in text:
                continue
            # c = list(map(lambda zs: f"{zs}", c))
            animation_data['x'].extend(x)
            animation_data['z'].extend(z)
            animation_data['y'].extend(y)
            animation_data['c'].extend(c)
            animation_data['subject_id'].extend([f"{enm}"] * len(x))
            animation_data['text'].extend(text)
            # st.write(z,x,y,c,text, color_mapping)# , color_mapping["0"])
            trace = go.Scatter3d(
                x=x, y=z, z=y,
                mode=mode_str, # "lines+markers",
                hovertext=text,
                hoverinfo="text",
                name=str(c[0]),
                legendgroup=str(c[0]),
                line=dict(
                    color=list(map(lambda pol: color_mapping[str(pol)], c)),
                    width=1,
                ),
                marker=dict(color=color_mapping[str(c[0])], size=4),
                opacity=0.7,
                showlegend=True if legend_track.get(str(c[0]), None) is None else False
            )
            traces.append(trace)
            legend_track[str(c[0])] = 1
        enm += 1
    animation_data = pd.DataFrame(animation_data)
    # animation_data['z'] = animation_data['z'].map(lambda x: round(x, 2))
    # animation_data = animation_data.replace({'Dimentia': 'Dementia'})
    ## -------------
    # x = results_data[input_parameter_name]['data'][input_color_column]
    # y = results_data[input_parameter_name]['data'][input_color_column].astype('category').cat.codes.map(lambda x: color_sequence[x])
    # color_mapping = dict(zip(list(x), list(y)))
    # age_list = results_data['age_list']
    # traces = []
    # legend_track = {}
    # enm = 1
    # animation_data = defaultdict(list)
    # for trace_input in trace_inputs:
    #     for z, x, y, _, c, text in trace_input:
    #         animation_data['x'].extend(x)
    #         animation_data['z'].extend(z)
    #         animation_data['y'].extend(y)
    #         animation_data['c'].extend(c)
    #         animation_data['text'].extend(text)
    #         trace = go.Scatter3d(
    #             x=x, y=z, z=y,
    #             mode="lines+markers",
    #             hovertext=text,
    #             hoverinfo="text",
    #             name = c[0],
    #             legendgroup=c[0],
    #             line=dict(
    #                 color=list(map(lambda x: color_mapping[x], c)),
    #                 width=1,
    #             ),
    #             marker=dict(color=color_mapping[c[0]], size=4),
    #             opacity=0.7,
    #             showlegend=True if legend_track.get(c[0], None) is None else False
    #         )
    #         traces.append(trace)
    #         legend_track[c[0]] = 1
    #     enm += 1
    # animation_data  = pd.DataFrame(animation_data)
    # animation_data['z'] = animation_data['z'].map(lambda x: round(x, 2))
    # animation_data = animation_data.replace({'Dimentia': 'Dementia'})
    
    if True:
        st.write('##### Trajectory')
        @st.cache
        def generate_plot():
            fig = go.Figure(data=traces)
            grid_color = 'white'
            fig.update_layout(
            width=1000,
            height=600,
            scene=dict(
            aspectratio = dict( x=0.5, y=1.25, z=0.5 ),
            yaxis_title="time-axis",
            xaxis_title="UMAP-X",
            zaxis_title="UMAP-Y",
            yaxis = dict(
                tickmode = 'array',
                tickvals = list(range(len(age_list_mapping))),
                ticktext =  age_list_mapping,
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            ),
            xaxis = dict(
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            ),
            zaxis= dict(
                backgroundcolor=grid_color,
                gridcolor=grid_color,
                showbackground=True,
                zerolinecolor='white',
                zerolinewidth=3
            )
            ),
            scene_camera=dict(eye=dict( x=0.5, y=0.8, z=0.75 )),
            autosize=False,
            )   
            return fig


        fig, color_bar = return_generated_figure(animation_data, 'c', title='', highlight_category=highlight_category)
        # st.write(fig)
        # st.info("Rendering trajectory plot will take few minutes....")
        if color_bar:
            col1, col2 = st.columns((4, 1))
            col1.plotly_chart(fig)
            col2.plotly_chart(color_bar)
        else:
            st.plotly_chart(fig)
        # st.plotly_chart(generate_plot())
    
    # st.write('### Change in lower dimensional space with time axis')
    # fig = px.scatter(animation_data, x="x", y="y", animation_frame="z", color="c", hover_name="text", color_discrete_sequence=px.colors.qualitative.G10)
    # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    # fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    # fig.update_layout(
    #     width=800,
    #     height=600,
    #     plot_bgcolor='white',
    # )
    # st.plotly_chart(fig)
