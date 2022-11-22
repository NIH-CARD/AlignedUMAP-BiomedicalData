#!/bin/bash

config_dir=$1
pip install omegaconf

submit_dir=${PWD}
echo ${PWD}
pwd
echo "Script Creation Started"
python create_scripts.py ${config_dir} "$2"
echo "Unlocking Directory Started"
# rm -r "${config_dir}"/tmp
mkdir -p "${config_dir}"/tmp
mkdir -p "${config_dir}"/log_scripts
snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}/execution_scripts" num_cores="$2" --unlock
# snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}" other_cmds="module load singularity \&\& export SINGULARITY_BIND='/data/dadua2,/data/CARD'" --unlock
echo "Snakemake Started"
snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}/execution_scripts" num_cores="$2" -j 64 --nolock
bash "${config_dir}"/final_script.sh
exit 0