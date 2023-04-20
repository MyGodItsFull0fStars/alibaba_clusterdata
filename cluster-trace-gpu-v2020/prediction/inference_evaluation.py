# %%
import torch
from os import walk
import pandas as pd

from typing import Any
from tqdm import tqdm
from timeit import default_timer as timer

from torch.utils.data import DataLoader
from lstm_models import UtilizationLSTM
from gpu_dataloader import UtilizationDataset

from utils import get_device, get_rmse, get_mape, symmetric_mean_absolute_percentage_error

# %%
device = get_device()
print(device)

# %%
def get_pytorch_files(model_dir: str) -> list[str]:
    model_list: list[str] = list()
    for (dir_path, _, file_names) in walk(model_dir):
        file_names = [f'{dir_path}/{file}' for file in file_names if file.endswith('.pt') and 'batch_size' not in file and 'rmse' not in file]
        model_list.extend(file_names)
    return model_list

# %%
model_list: list[str] = get_pytorch_files('models/')
print(model_list)

# %%
def get_model_params_from_path(model_path: str) -> dict[str, Any]:
    if torch.cuda.is_available():
        return torch.load(model_path)
    else:
        return torch.load(model_path, map_location=torch.device('cpu'))

# %%
def get_hyperparams_from_model_params(model_params: dict[str, Any]) -> tuple[str, int, int, int, int, UtilizationLSTM]:
    name: str = model_params['name']
    input_size: int = model_params['input_size']
    hidden_size: int = model_params['hidden_size']
    num_layers: int = model_params['num_layers']
    num_classes: int = model_params['num_classes']
    
    model = UtilizationLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(model_params['model_state_dict'])
    model.eval()
    
    return name, input_size, hidden_size, num_layers, num_classes, model

# %%
small_df: bool = True

# %%
print('create datasets')
no_tasks_dataset = UtilizationDataset(is_training=True, small_df=small_df, include_tasks=False, include_instance=False)
with_tasks_dataset = UtilizationDataset(is_training=True, small_df=small_df, include_tasks=True, include_instance=False)
instance_pmse_dataset = UtilizationDataset(is_training=True, small_df=small_df, include_tasks=True, include_instance=True)

# %%
batch_sizes: list[int] = [bs for bs in range(100, 2001, 100)]

# %%
def get_dataset(model_name: str) -> UtilizationDataset:
    if 'without_tasks' in model_name:
        return no_tasks_dataset
    elif 'with_tasks' in model_name:
        return with_tasks_dataset
    else:
        return instance_pmse_dataset

# %%
for model_path in model_list:
    model_params: dict[str, Any] = get_model_params_from_path(model_path)
    name, input_size, hidden_size, num_layers, num_classes, model = get_hyperparams_from_model_params(model_params)
    print(f'Starting model {name}')
    data_set = get_dataset(name)
    
    model_df: pd.DataFrame = pd.DataFrame(index=batch_sizes, columns=['Total Time', 'Average Time'])
    
    for bs in batch_sizes:
        data_loader = DataLoader(dataset=data_set, batch_size=bs, shuffle=False, num_workers=10)

        inference_times: list[float] = list()
        
        for _, (inputs, labels) in enumerate(tqdm(data_loader, leave=False)):
            # send input and label to device
            inputs, labels = inputs.to(device), labels.to(device)
            start = timer()
            # forward input to model
            predictions = model(inputs).to(device)
            end = timer()
            inference_times.append(end - start)
    
        total_time = sum(inference_times)
        average_time = total_time / len(inference_times)
        
        model_df.loc[bs] = total_time, average_time
        
    model_df.to_csv(f'./evaluation/inference/{name}_inference.csv')
    
    print(f'Finished model: {name}')


