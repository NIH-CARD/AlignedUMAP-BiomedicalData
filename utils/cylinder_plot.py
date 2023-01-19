import copy
import random

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import interpolate


@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
def get_cylinder_plot(selected_data_dummy, input_color_column, color_mapping_original, height=1000, width=1300):
    selected_feature = "x"
    selected_feature2 = "y"
    selected_age = "z"
    plot_type = "3D"
    columns_mapping = {
        selected_age: 'days from baseline',  # x-axis
        selected_feature: 'Median value1',  # y-axis
        selected_feature2: 'Median value2',  # z-axis
        'subject_id': 'PG.Genes',
        input_color_column: 'STUDY_DIAGNOSIS'
    }
    selected_data = selected_data_dummy.copy().rename(columns=columns_mapping)

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
                x_temp = [x_surface[i] - k * x_surface_std[i], x_surface[i] - k * x_surface_std[i], x_surface[i] + k * x_surface_std[i],
                          x_surface[i] + k * x_surface_std[i]]
                y_temp = [y_surface[i] - k * y_surface_std[i], y_surface[i] + k * y_surface_std[i], y_surface[i] + k * y_surface_std[i],
                          y_surface[i] - k * y_surface_std[i]]
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
        color_list = list(px.colors.qualitative.G10)
        color_mapping = {}
        for e, gene in enumerate(selected_data['STUDY_DIAGNOSIS'].unique()):
            color_mapping[gene] = color_list[e % 10]

        temp = selected_data[selected_data['PG.Genes'].isin(background_genes)]
        new_color_mapping = {}
        for e in range(len(selected_data)):
            new_color_mapping[selected_data.iloc[e]['PG.Genes']] = color_mapping[
                selected_data.iloc[e]['STUDY_DIAGNOSIS']]
        if 'Median value1' in temp.columns:
            scatter_data = []
            flag = {}
            colorlist = px.colors.qualitative.Dark24
            for e, dia in enumerate(list(temp['STUDY_DIAGNOSIS'].unique())):
                V = temp[(temp['STUDY_DIAGNOSIS'] == dia)]
                V = V.sort_values('days from baseline')
                V_std = V.groupby('days from baseline').agg('std').dropna(subset=['Median value1', 'Median value2']).sort_index()
                V = V.groupby('days from baseline').agg('mean').reset_index().sort_values(by='days from baseline')
                V = V[V['days from baseline'].isin(list(V_std.index))]
                x = V['days from baseline'].values
                y = V['Median value1'].values
                z = V['Median value2'].values
                c = color_mapping_original.get(dia, colorlist[(e + 2) % len(set(temp['STUDY_DIAGNOSIS']))])  # 'blue' if e==0 else 'red'
                colorscale = [[0, c], [1, c]]
                enm = 0
                opacity_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1, 1]
                opacity_list = [0.6, 0.4, 0.2, 0.1]
                opacity_list = [0.5, 0.2, 0.1, 0.01, 0.01]
                opacity_list = [0.1, 0.05, 0.02, 0.01, 0.01]
                for x_new, y_new, z_new in surface_ellipse_params(V, V_std):
                    cyl1 = go.Surface(x=x_new, y=y_new, z=z_new, colorscale=colorscale, showscale=False, opacity=opacity_list[enm], name=dia, legendgroup=dia, showlegend=False if enm == 0 else False)
                    scatter_data.append(cyl1)
                    enm += 1
                line1 = go.Scatter3d(x=x, y=y, z=z, marker=dict(size=5, color=c, line=dict(color=c, width=8)), line=dict(color=c, width=8), name=dia, legendgroup=dia, showlegend=True)  # colorscale='Viridis',
                scatter_data.append(cyl1)
                scatter_data.append(line1)
        else:
            scatter_data = []
            flag = {}
            for gene in list(set(background_genes)):
                temp_new = temp[temp['PG.Genes'] == gene]
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

    def generate_plotly_plot(selected_data, background_genes, selected_genes, plot_type='2D'):
        temp = selected_data[selected_data['PG.Genes'].isin(background_genes + selected_genes)]
        color_list = list(px.colors.qualitative.G10)
        color_mapping = {}
        for e, gene in enumerate(temp['STUDY_DIAGNOSIS'].unique()):
            color_mapping[gene] = color_list[e % 10]

        new_color_mapping = {}
        for e in range(len(temp)):
            new_color_mapping[temp.iloc[e]['PG.Genes']] = color_mapping[temp.iloc[e]['STUDY_DIAGNOSIS']]

        new_line_mapping = {i: 0.5 if i not in selected_genes else 5 for i in temp['PG.Genes']}
        new_opacity_mapping = {i: 0.2 if i not in selected_genes else 1 for i in temp['PG.Genes']}
        new_hover_mapping = {temp.iloc[i]['PG.Genes']: None for i in range(len(temp['PG.Genes']))}
        scatter_data = generate_plotly_background(selected_data.copy(), copy.deepcopy(selected_genes),
                                                  copy.deepcopy(background_genes), copy.deepcopy(new_color_mapping),
                                                  copy.deepcopy(new_line_mapping), copy.deepcopy(new_opacity_mapping),
                                                  copy.deepcopy(new_hover_mapping), selected_age)
        return scatter_data

    fraction_subjects = 1.0
    selected_genes = []
    random.seed(42)
    l = list(selected_data['PG.Genes'].unique())
    random.shuffle(l)
    background_genes = list(set(l[:int(len(l) * fraction_subjects)]) - set(selected_genes))
    alldata = generate_plotly_plot(selected_data, background_genes, selected_genes, plot_type=plot_type)
    fig = go.Figure(
        data=alldata,
        layout=go.Layout(height=650, width=1000)
    )
    if plot_type == '2D':
        fig.update_layout(
            showlegend=True,
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
    return fig
