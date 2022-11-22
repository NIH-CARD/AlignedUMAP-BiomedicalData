import plotly.express as px
import numpy as np
import pandas as pd
import json
import numpy as np
import copy
import platform
if '3.7' in platform.python_version():
    import pickle5  as pickle
else:
    import pickle

import streamlit as st
import plotly.graph_objects as go
import base64
from collections import defaultdict

@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
def get_cylinder_plot(selected_data_dummy, input_color_column, color_mapping_original, height=1000, width=1300):
    selected_feature = "x" #  x=x, y=z, z=y,
    selected_feature2 = "y"
    selected_age = "z"
    plot_type = "3D"
    columns_mapping = {
        selected_age: 'days from baseline', # x-axis
        selected_feature: 'Median value1', # y-axis
        selected_feature2: 'Median value2', # z-axis
        'subject_id': 'PG.Genes',
        input_color_column: 'STUDY_DIAGNOSIS'
    }
    # selected_data = results_data[input_parameter_name]['data'].copy()
    selected_data = selected_data_dummy.copy().rename(columns=columns_mapping)
    import scipy.stats
    import numpy as np
    import math
    import random


    def get_95ci(x):
        random.seed(42)
        data = list(x)
        # v = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
        v = scipy.stats.norm.interval(alpha=0.95, loc=np.mean(data), scale=scipy.stats.sem(data))
        return (v[-1] - v[0]) / 2


    from scipy import interpolate


    def surface_ellipse_params(V, V_std):
        z_surface = V['days from baseline'].values
        x_surface = V['Median value1'].values
        y_surface = V['Median value2'].values
        x_surface_std = V_std['Median value1'].values
        y_surface_std = V_std['Median value2'].values
        z_new = []
        x_new = []
        y_new = []
        for k in [0.5, 1, 2, 3]:
            for i in range(0, len(z_surface)):
                x_surface_std[i] = max(x_surface_std[i], 1e-5)
                y_surface_std[i] = max(y_surface_std[i], 1e-5)
                z_new.append([z_surface[i]] * 100)
                x_temp = [x_surface[i] - k * x_surface_std[i], x_surface[i] - k * x_surface_std[i], x_surface[i] + k*x_surface_std[i],
                          x_surface[i] + k*x_surface_std[i]]
                y_temp = [y_surface[i] - k*y_surface_std[i], y_surface[i] + k*y_surface_std[i], y_surface[i] + k*y_surface_std[i],
                          y_surface[i] - k*y_surface_std[i]]
                # print (x_temp)
                # print (y_temp)
                tck, u = interpolate.splprep([x_temp + x_temp[:1], y_temp + y_temp[:1]], s=0, per=True)
                unew = np.linspace(0, 1, 100)
                basic_form = interpolate.splev(unew, tck)
                x_new.append(basic_form[0])
                y_new.append(basic_form[1])
            yield z_new, x_new, y_new


    def generate_plotly_background(selected_data, selected_genes, background_genes, new_color_mapping, new_line_mapping,
                                   new_opacity_mapping, new_hover_mapping, selected_age):
        layout = go.Layout(
            autosize=False,
            width=900,
            height=600)
        # st.write(selected_data.head())
        # fig = go.Figure(layout=layout)
        color_list = list(px.colors.qualitative.G10)
        color_mapping = {}
        for e, gene in enumerate(selected_data['STUDY_DIAGNOSIS'].unique()):
            color_mapping[gene] = color_list[e % 10]

        temp = selected_data[selected_data['PG.Genes'].isin(background_genes)]
        new_color_mapping = {}
        for e in range(len(selected_data)):
            new_color_mapping[selected_data.iloc[e]['PG.Genes']] = color_mapping[
                selected_data.iloc[e]['STUDY_DIAGNOSIS']]
        # new_color_mapping = {i: color_mapping[i] for e, i in enumerate(temp['STUDY_DIAGNOSIS'].unique())}
        # st.write(temp.columns)
        if 'Median value1' in temp.columns:
            scatter_data = []
            flag = {}
            # if temp['days from baseline'].max() > 200:
            #     temp['days from baseline'] = temp['days from baseline'] // 360
            # colorlist = ['blue', 'red', 'orange', 'green', 'yellow', 'brown', 'violet'] * 2
            colorlist = px.colors.qualitative.Dark24
            for e, dia in enumerate(list(temp['STUDY_DIAGNOSIS'].unique())):
                V = temp[(temp['STUDY_DIAGNOSIS'] == dia)]
                V = V.sort_values('days from baseline')
                # V['Median value1'] = V['Median value1'].rolling(window=20, min_periods=1).mean()
                # V['Median value2'] = V['Median value2'].rolling(window=20, min_periods=1).mean()
                # V_std = V.groupby('age_round').agg(get_95ci).fillna(0).sort_index()
                # V_std = V.groupby('days from baseline').agg(get_95ci).dropna(subset=['Median value1', 'Median value2']).sort_index()
                V_std = V.groupby('days from baseline').agg('std').dropna(subset=['Median value1', 'Median value2']).sort_index()
                V = V.groupby('days from baseline').agg('mean').reset_index().sort_values(by='days from baseline')
                V = V[V['days from baseline'].isin(list(V_std.index))]
                x = V['days from baseline'].values
                y = V['Median value1'].values
                z = V['Median value2'].values
                # c = colorlist[e]  # 'blue' if e==0 else 'red'
                c = color_mapping_original.get(dia, colorlist[(e+2) % len(set(temp['STUDY_DIAGNOSIS']))])  # 'blue' if e==0 else 'red'
                colorscale = [[0, c], [1, c]]
                enm = 0
                opacity_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1, 1]
                opacity_list = [0.6, 0.4, 0.2, 0.1]
                opacity_list = [0.5, 0.2, 0.1, 0.01, 0.01]
                opacity_list = [0.1, 0.05, 0.02, 0.01, 0.01]
                for x_new, y_new, z_new in surface_ellipse_params(V, V_std):
                    cyl1 = go.Surface(x=x_new, y=y_new, z=z_new, colorscale=colorscale, showscale=False, opacity=opacity_list[enm],name=dia, legendgroup=dia, showlegend=False if enm==0 else False)
                    scatter_data.append(cyl1)
                    enm += 1
                # cyl1 = go.Volume(x=x_new, y=y_new, z=z_new, colorscale=colorscale, showscale=False, opacity=0.5,name=dia, legendgroup=dia, showlegend=True)
                line1 = go.Scatter3d(x=x, y=y, z=z, marker=dict(size=5, color=c, line=dict(color=c,width=8)), line=dict( color=c, width=8, ), name=dia, legendgroup=dia, showlegend=True) # colorscale='Viridis',
                scatter_data.append(cyl1)
                scatter_data.append(line1)
        else:
            scatter_data = []
            flag = {}
            for gene in list(set(background_genes)):
                temp_new = temp[temp['PG.Genes'] == gene]
                # temp['STUDY_DIAGNOSIS'].iloc[-1]
                scatter_data.append(go.Scatter(x=temp_new['days from baseline'], y=temp_new['Median value'].values,
                                               mode='markers+lines',

                                               legendgroup=temp_new['STUDY_DIAGNOSIS'].iloc[-1],
                                               legendgrouptitle_text=temp_new['STUDY_DIAGNOSIS'].iloc[-1],
                                               name=temp_new['STUDY_DIAGNOSIS'].iloc[-1],

                                               line_color=new_color_mapping[gene], line_width=new_line_mapping[gene],
                                               hoverinfo='text', hovertext=new_hover_mapping[gene],
                                               marker=dict(size=3, color=new_color_mapping[gene],
                                                           opacity=new_opacity_mapping[gene]),
                                               showlegend=True if temp_new['STUDY_DIAGNOSIS'].iloc[
                                                                      -1] not in flag else False, ))
                flag[temp_new['STUDY_DIAGNOSIS'].iloc[-1]] = True
        return scatter_data


    # @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def generate_plotly_plot(selected_data, background_genes, selected_genes, plot_type='2D'):

        temp = selected_data[selected_data['PG.Genes'].isin(background_genes + selected_genes)]
        # st.write(temp.head())

        # Create traces
        color_list = list(px.colors.qualitative.G10)
        color_mapping = {}
        for e, gene in enumerate(temp['STUDY_DIAGNOSIS'].unique()):
            color_mapping[gene] = color_list[e % 10]

        new_color_mapping = {}
        for e in range(len(temp)):
            new_color_mapping[temp.iloc[e]['PG.Genes']] = color_mapping[temp.iloc[e]['STUDY_DIAGNOSIS']]

        # color_mapping = {}
        # for e, gene in enumerate(selected_genes):
        #     color_mapping[gene] = color_list[e%10]

        # # new_color_mapping = {i: 'lightpink' if i not in selected_genes else 'dodgerblue' for e, i in enumerate(temp['PG.Genes'].unique())}
        # new_color_mapping = {i: 'lightgrey' if i not in selected_genes else color_mapping[i] for e, i in enumerate(temp['PG.Genes'].unique())}
        new_line_mapping = {i: 0.5 if i not in selected_genes else 5 for i in temp['PG.Genes']}
        new_opacity_mapping = {i: 0.2 if i not in selected_genes else 1 for i in temp['PG.Genes']}
        # new_hover_mapping = {temp.iloc[i]['PG.Genes']: None if temp.iloc[i]['PG.Genes'] not in selected_genes else '<br>'.join(list(temp.iloc[i][['PG.Genes', 'PG.ProteinAccessions', 'PG.ProteinDescriptions']])) for i in range(len(temp['PG.Genes']))}
        new_hover_mapping = {temp.iloc[i]['PG.Genes']: None for i in range(len(temp['PG.Genes']))}

        scatter_data = generate_plotly_background(selected_data.copy(), copy.deepcopy(selected_genes),
                                                  copy.deepcopy(background_genes), copy.deepcopy(new_color_mapping),
                                                  copy.deepcopy(new_line_mapping), copy.deepcopy(new_opacity_mapping),
                                                  copy.deepcopy(new_hover_mapping), selected_age)
        return scatter_data
        layout = go.Layout(
            autosize=False,
            width=900,
            height=600
        )
        if not 'Median value' in temp.columns:
            if len(selected_genes) == 0:
                return scatter_data

            selected_scatter_data = []
            # if temp['days from baseline'].max() > 200:
            #     temp['days from baseline'] = temp['days from baseline'] // 360
            for gene in list(set(selected_genes)):
                temp_new = temp[temp['PG.Genes'] == gene]
                selected_scatter_data.append(
                    go.Scatter3d(x=temp_new['days from baseline'], y=temp_new['Median value1'].values,
                                 z=temp_new['Median value2'].values,
                                 mode='markers+lines',
                                 legendgroup=gene if gene in selected_genes else 'others',
                                 legendgrouptitle_text=gene if gene in selected_genes else 'others',
                                 name=gene.split(';')[0] if gene in selected_genes else 'others',
                                 line_color=new_color_mapping[gene], line_width=new_line_mapping[gene],
                                 hoverinfo='text', hovertext=new_hover_mapping[gene],
                                 marker=dict(size=10, color=new_color_mapping[gene], opacity=new_opacity_mapping[gene]),
                                 showlegend=True if gene in selected_genes else False, ))
            return scatter_data + selected_scatter_data

        else:
            if len(selected_genes) == 0:
                return scatter_data

            selected_scatter_data = []
            for gene in list(set(selected_genes)):
                temp_new = temp[temp['PG.Genes'] == gene]
                selected_scatter_data.append(
                    go.Scatter(x=temp_new['days from baseline'], y=temp_new['Median value'].values,
                               mode='markers+lines',
                               legendgroup=gene if gene in selected_genes else 'others',
                               legendgrouptitle_text=gene if gene in selected_genes else 'others',
                               name=gene.split(';')[0] if gene in selected_genes else 'others',
                               line_color=new_color_mapping[gene], line_width=new_line_mapping[gene],
                               hoverinfo='text', hovertext=new_hover_mapping[gene],
                               marker=dict(size=10, color=new_color_mapping[gene], opacity=new_opacity_mapping[gene]),
                               showlegend=True if gene in selected_genes else False, ))

        return scatter_data + selected_scatter_data


    if True:  # st.expander("", expanded=True): # with st.expander("", expanded=True): #
        # selected_genes = st.sidebar.multiselect('Select multiple subjects', selected_data['PG.Genes'].unique(), default=['ADNI-002_S_0295'])
        # selected_genes = st.sidebar.multiselect('Select multiple subjects', selected_data['PG.Genes'].unique(), default=['ADNI-002_S_0619'])
        fraction_subjects = 1.0 # st.sidebar.slider('Fraction of subject to be shown', 0.0, 1.0,
                                              # 0.2 if plot_type == '2D' else 1.0, 0.1)
        # if len(selected_genes) == 0:
        # selected_genes = ['ADNI-002_S_0295']
        #     selected_genes = ['ADNI-002_S_0619']
        selected_genes = []
        #  st.write(selected_genes)
        # cols = st.columns([3, 1])
        if True:  # len(selected_genes) > 0: #and plot_type == '2D':
            random.seed(42)
            l = list(selected_data['PG.Genes'].unique())
            random.shuffle(l)
            # temp = selected_data[selected_data['PG.Genes'].isin(selected_genes + l[:int(len(l) * 0.01)])]
            background_genes = list(set(l[:int(len(l) * fraction_subjects)]) - set(selected_genes))
            alldata = generate_plotly_plot(selected_data, background_genes, selected_genes, plot_type=plot_type)
            fig = go.Figure(
                data=alldata,
                layout=go.Layout(height=650, width=1000)
            )
            if plot_type == '2D':

                # fig.update_yaxes(type="log")
                fig.update_layout(
                    showlegend=True,
                    # title="Plot Title",
                    xaxis_title=selected_age,
                    yaxis_title=selected_feature,
                    legend_title="Selected Participants",
                    plot_bgcolor="white",
                    font=dict(
                        family="Courier New, monospace",
                        size=18, )
                )
            else:
                pass
                # fig.update_layout(scene_camera_eye_z=0.55, height=height, width=width)
                # fig.update_layout(scene_camera=dict(eye=dict( x=0.5, y=0.8, z=0.75 )), height=height, width=width,)
                # aspectratio = dict( x=0.5, y=1.25, z=0.5 ),
                # fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=2, y=1, z=1))
                # fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1.25, y=0.5, z=0.5 ))
                # fig.layout.scene.camera.projection.type = "orthographic"
                # fig.update_layout(
                #     showlegend=True,
                #     # title="Plot Title",
                #     scene=dict(
                #         xaxis_title=selected_age,
                #         yaxis_title=selected_feature,
                #         zaxis_title=selected_feature2
                #     ),
                #     legend_title="Selected Participants",
                #     plot_bgcolor="white",
                #     font=dict(
                #         family="Courier New, monospace",
                #         size=12, )
                # )
                # fig.update_traces(showlegend=True)
            # fig.update_xaxes(showgrid=False)
            # fig.update_yaxes(type="log", range = [2, 6], showgrid=False)
            # fig.update_layout(
            #     hoverlabel=dict(
            #         font_size=12,
            #     )
            # )
            # fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
            # cols[0].write(fig, use_container_width=True, centering=False)
    return fig