from utils.cylinder_plot import get_cylinder_plot
import pickle
import streamlit as st
from collections import defaultdict
import plotly.io as pio
pio.templates.default = pio.templates["plotly_white"] # "plotly_white"
SPECS = {}

def app(fname, input_dataset_name, color_column_list, metadata_descriptions, combined_best_parameters, **kwargs):
    if not input_dataset_name in SPECS:
        SPECS[input_dataset_name] = {
            'dataset_name': input_dataset_name,
            'colorname': color_column_list[0],
            'num_subjects': 1,
            'selected_cats': [],
            'mapping_legend': {},
            'annotation_legend': {},
            'time_mapping': ['time', { }],
            'alpha': 1
        }
    color_mapping = {}
    time_axis_mapping = {}

    input_color_column = st.sidebar.selectbox('Select the color factor', color_column_list, index=0)
    st.sidebar.info("***{}***: {}".format(input_color_column, metadata_descriptions.get(input_color_column, 'Self Explanatory')))

    fpath = fname # f"results_data/{input_dataset_name}/umap_aligned/generated_data/{input_dataset_name}_16.pickle"
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
