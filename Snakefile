import os
from pathlib import Path
dir_path=Path(config['pwd_path'])
exec_repath=Path(config['exec_repath'])
num_cores = int(config['num_cores'])
str_dir_path=str(dir_path)
str_exec_path=str(exec_repath)
os.makedirs(dir_path / exec_repath / 'tmp', exist_ok=True)
os.makedirs(dir_path /  exec_repath / 'log_scripts', exist_ok=True)
all_outputs = []
for i in os.listdir(dir_path / exec_repath):
    if (i[-3:] == '.sh') and (not os.path.exists(str(dir_path / exec_repath / 'tmp' / f"{i.split('.sh')[0]}.running"))) :
        all_outputs.append(    str(dir_path / exec_repath / 'tmp' / f"{i.split('.sh')[0]}.out")   )

print (str(str_dir_path +  '/' + str_exec_path + "/tmp"))
print (all_outputs)
rule all:
    input:
        all_outputs

rule compute1:
    output:
        touch(str_dir_path + '/' + str_exec_path + "/tmp/{script_name}.out")
    resources: mem_gb=128, cpus=num_cores, time_minutes=240
    threads: num_cores
    params:
        output_dir=str_dir_path + '/' +  str_exec_path + "/log_scripts"
    shell:
        '''
            cd {str_dir_path} &&
            touch ./{str_exec_path}/tmp/{wildcards.script_name}.running &&
            chmod +x ./{str_exec_path}/{wildcards.script_name}.sh &&
            ./{str_exec_path}/{wildcards.script_name}.sh &&
            rm {str_dir_path}/{str_exec_path}/tmp/{wildcards.script_name}.running
        '''