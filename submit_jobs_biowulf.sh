#!/bin/bash

#SBATCH --partition=norm
#SBATCH --time=360
#SBATCH --cpus-per-task=16
#SBATCH --mem=240g
#SBATCH --output=logs/myjob_%j.out
#SBATCH --error=logs/myjob_%j.err

# config_dir="configs/NMOCA"
# config_dir="configs/DIAGNOSIS"
config_dir=$1
module load snakemake
module load python
pip install omegaconf

submit_dir=${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
cd ${SLURM_SUBMIT_DIR}
echo "Script Creation Started"
python create_scripts.py ${config_dir} "$2"
pwd
echo "Unlocking Directory Started"
# rm -r "${config_dir}"/tmp
mkdir -p "${config_dir}"/tmp
mkdir -p "${config_dir}"/log_scripts

snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}/execution_scripts" num_cores="$2" --unlock
# snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}" other_cmds="module load singularity \&\& export SINGULARITY_BIND='/data/dadua2,/data/CARD'" --unlock
echo "Snakemake Started"
snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}/execution_scripts" num_cores="$2" --profile su-cpu --nolock
bash "${config_dir}"/final_script.sh
exit 0

# snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}" num_cores="$2"--unlock
# snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}" other_cmds="module load singularity \&\& export SINGULARITY_BIND='/data/dadua2,/data/CARD'" --unlock
# echo "Snakemake Started"
# snakemake --config pwd_path="${submit_dir}" exec_repath="${config_dir}" num_cores="$2" --profile su-cpu --nolock
# bash "${config_dir}"/execution_scripts/final_script.sh
# exit 0