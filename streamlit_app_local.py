import os
import sys
import streamlit as st
st.set_page_config(layout="wide")
import plotly.io as pio
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
import pickle
pio.templates.default = pio.templates["plotly_white"] # "plotly_white"
config = OmegaConf.load(sys.argv[1])
input_dataset_name = config['dataset_name'].split('.csv')[0]
if config['metadata_name'] == "":
    data = pd.read_csv(Path(config['data_dir']) / f"{input_dataset_name}.csv")
    metadata = pd.DataFrame({'subject_id':data.reset_index()['subject_id'].unique()})
    metadata['color'] = 'NoColor'
    metadata = metadata.set_index('subject_id')
else:
    metadata = pd.read_csv(Path(config['data_dir']) / config['metadata_name']).set_index('subject_id')

st.header("Aligned-UMAP for Longitudinal Biomedical Datasets (Local Version)")
cols = st.columns(3)
input_visualization_method = "umap_aligned"
num_cores = config['num_cores']
result_dir = Path(config['result_dir'])
if num_cores == -1:
    num_cores = os.cpu_count()
sample_fraction = config['sample_fraction']
sample_fraction = str(float(sample_fraction)) if not sample_fraction == 1 else "1.0"
sample_text = f"_{sample_fraction}"

info_msg1 = st.sidebar.empty()
info_msg2 = st.sidebar.empty()
dataset_p = f"results_data/{input_dataset_name}/{input_visualization_method}/generated_data/"
fname = result_dir / f"{input_dataset_name}/{input_visualization_method}/generated_data/{input_dataset_name}_{num_cores}{sample_text}.pickle"
if not os.path.exists(fname):
    st.stop()
else:
    info_msg2.success("Click tabs to visualize longitudinal patterns.")

color_column_list = list(metadata.columns)
from streamlit_apps_local import browsable_component, select, parameter_effect
from streamlit_multiapp import MultiApp
app = MultiApp()

with open(fname, "rb") as f:
    S = pickle.load(f)

parameter_fix = S['complete_dataframe']['input_parameter_id'].iloc[0]
combined_best_parameters = {
        input_dataset_name: [[1.25, -1.25, -1.25], parameter_fix],
}
metadata_descriptions = {}
params = { "input_dataset_name": input_dataset_name, "color_column_list": color_column_list, "fname": fname,
           "combined_best_parameters": combined_best_parameters, 'metadata_descriptions': metadata_descriptions }
app.add_app("Home", select.app, params)
app.add_app("Browse Trajectory Plots", browsable_component.app, params)
app.add_app("Aligned UMAP Parameter Tuning", parameter_effect.app, params)
app.run()