# %%
from os import walk
import subprocess
import parser_arguments

# %%
model_configuration_path: str = './model_configs'

python_script: str = 'lstm_timestamp_prediction.py'
dry_run_flag: str = '--dry-run' if parser_arguments.dry_run else '--no-dry-run'

# %%
config_files: list[str] = list()


# %%
for (dir_path, dir_names, file_names) in walk(model_configuration_path):
    print(dir_path, dir_names, file_names)
    file_names = [f'{dir_path}/{file}' for file in file_names]
    config_files.extend(file_names)


# %%
for config in config_files:
    subprocess.run(['python', python_script,
                   f'--config={config}', dry_run_flag])
    print(f'finished {config}')

# %%
print('done')
