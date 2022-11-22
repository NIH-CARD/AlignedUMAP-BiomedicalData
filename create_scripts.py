import os
import sys
from omegaconf import OmegaConf
from pathlib import Path

config_fname = Path(sys.argv[1]) / 'configs.yaml'
config = OmegaConf.load(config_fname)
python_path = config['python_path']
if config['visualization_method'] == 'umap_aligned':
    num_parameters = len(config["metric"]) * len(config["alignment_regularisation"])
    num_parameters *= len(config["alignment_window_size"]) * len(config["n_neighbors"]) * len(config["min_dist"])
    template = f"{python_path} apply_model.py {config['dataset_name']} {config['metadata_name']}"
else:
    num_parameters = len(config["metric"]) * len(config["n_neighbors"]) * len(config["min_dist"])
    template = f"{python_path} apply_model_umap.py {config['dataset_name']} {config['metadata_name']}"

if len(sys.argv) >= 4:
    sample_text = f" {sys.argv[3]}"
else:
    sample_text = ""

os.makedirs(f"{sys.argv[1]}/execution_scripts", exist_ok=True)
for i in range(num_parameters):
    st = template + f" {i} {str(config_fname)} {sys.argv[2]}{sample_text}"
    with open(f'{sys.argv[1]}/execution_scripts/script{i}.sh', 'w') as f:
        f.write(st)

    os.system(f'chmod +x {sys.argv[1]}/execution_scripts/script{i}.sh')

if config['visualization_method'] == 'umap_aligned':
    template = f"{python_path} apply_model.py {config['dataset_name']} {config['metadata_name']} {num_parameters} {str(config_fname)} {sys.argv[2]}{sample_text}"
else:
    template = f"{python_path} apply_model_umap.py {config['dataset_name']} {config['metadata_name']} {num_parameters} {str(config_fname)} {sys.argv[2]} {sample_text}"

with open(f'{sys.argv[1]}/final_script.sh', 'w') as f:
    f.write(template)

os.system(f'chmod +x {sys.argv[1]}/final_script.sh')