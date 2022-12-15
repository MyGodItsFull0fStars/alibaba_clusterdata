# %%
print('import libraries')
import time

from typing import List

import torch
import torch.nn as nn

from lstm_models import LSTM, UtilizationLSTM
from gpu_dataloader import MachineDatasetContainer, MachineDataset
from torch.utils.data import DataLoader

# used for the dataframes
import pandas as pd
from tqdm import tqdm

import yaml
from yaml.loader import SafeLoader

from utils import get_device, get_rmse, get_mae


# %%
print('load yaml config file')
with open('./model_configs/machine_test/machine_test.yaml') as file:
    yaml_config = yaml.load(file, Loader=SafeLoader)

# %%
batch_size: int = yaml_config['dataset']['batch_size']
small_df: bool = yaml_config['dataset']['small_df']
include_tasks: bool = yaml_config['dataset']['include_tasks']

# %%
print('load datasets')
dataset = MachineDatasetContainer(is_training=True, small_df=small_df, include_tasks=include_tasks)
test_dataset = MachineDatasetContainer(is_training=False, small_df=small_df, include_tasks=include_tasks)

# %%
num_epochs: int = yaml_config['model']['num_epochs']
learning_rate: float = yaml_config['model']['learning_rate']

input_size: int = dataset.get_model_input_size()
hidden_size: int = yaml_config['model']['hidden_size']
num_layers: int = yaml_config['model']['num_layers']
num_classes: int = dataset.get_model_num_classes()

print(f'''
      input size: {input_size}
      hidden size: {hidden_size}
      num layers: {num_layers}
      num epochs: {num_epochs}
      ''')

device = get_device()

INCLUDE_WANDB: bool = False

# %%
if INCLUDE_WANDB == True:
    print('init wandb')
    import wandb
    wandb.init(project=yaml_config['model']['name'])

    wandb.config.num_epochs = num_epochs
    wandb.config.learning_rate = learning_rate
    wandb.config.input_size = input_size
    wandb.config.hidden_size = hidden_size
    wandb.config.num_layers = num_layers
    wandb.config.num_classes = num_classes

# %%
LOSS: str = 'loss'
RMSE_TRAINING: str = 'root mean squared error (training)'
MAE_TRAINING: str = 'mean absolute error (training)'

if INCLUDE_WANDB:
    wandb.define_metric(LOSS, summary='min')
    wandb.define_metric(RMSE_TRAINING, summary='min')
    wandb.define_metric(MAE_TRAINING, summary='min')

# %%
print('init lstm model')
model = UtilizationLSTM(num_classes, input_size, hidden_size, num_layers)
model.train()

if INCLUDE_WANDB:
    wandb.watch(model)

# %%
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler_config = yaml_config['model']['scheduler']
patience = scheduler_config['patience']
factor = scheduler_config['factor']
min_lr = scheduler_config['min_lr']
eps = scheduler_config['eps']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr, eps=eps)
del patience, factor, min_lr, eps

# %%
def log_training_metrics(predictions, labels, loss):
    # logging to wandb
    if torch.cuda.is_available():
        o = predictions.cpu().detach().numpy()
        l = labels.cpu().detach().numpy()
    else:
        o = predictions.detach().numpy()
        l = labels.detach().numpy()
    rmse = get_rmse(o, l)
    mae = get_mae(o, l)
    log_dict: dict = {
        LOSS: loss.item(),
        RMSE_TRAINING: rmse,
        MAE_TRAINING: mae,
    }
    wandb.log(log_dict)

# %%
def convert_datasets_to_data_loaders(data: MachineDatasetContainer, shuffle: bool = False, num_workers: int = 5) -> List[DataLoader]:
    data_loaders: List[DataLoader] = list()
    for dataset in data.dataset_list:
        data_loaders.append(
            DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        )
        
    return data_loaders

print('init dataloaders')
train_data_loaders: List[DataLoader] = convert_datasets_to_data_loaders(dataset)
test_data_loaders: List[DataLoader] = convert_datasets_to_data_loaders(test_dataset)

# %%
if torch.has_cuda:
    print('empty cuda cache')
    torch.cuda.empty_cache()

# %%
def train_loader_training_loop(train_loader: DataLoader) -> float:
    predictions, labels, loss = 0, 0, 0
    
    for _, (inputs, labels) in enumerate(train_loader):
        # send input and label to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward input to model
        predictions = model(inputs).to(device)

        optimizer.zero_grad()
        loss = criterion(predictions, labels)
        # backward propagation
        loss.backward()
        # update weights
        optimizer.step()
        
    if INCLUDE_WANDB:
        log_training_metrics(predictions, labels, loss)
        
    return loss.item()
            

# %%
def validation_loop():
    with torch.no_grad():
        for test_loader in test_data_loaders:
            for _, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                val_prediction = model(inputs).to(device)
                val_loss = criterion(val_prediction, labels)
                scheduler.step(val_loss)

# %%
loss_val = None
loss_progression: list = list()

print('start training loop')
for epoch in (pbar := tqdm(range(0, num_epochs), desc=f'Training Loop (0) -- Loss: {loss_val}')):
    
    for train_loader in train_data_loaders:
        
        loss_val = train_loader_training_loop(train_loader)
        loss_progression.append(loss_val)
        
        pbar.set_description(f'Training Loop ({epoch + 1}) -- Loss: {loss_val:.5f}')
        
    validation_loop()
                

# %%
current_time = time.ctime()

# %%
model.eval()

if yaml_config['model']['save_model'] and False:
    model_name = f'models/epochs-{num_epochs}-{current_time}'
    torch.save(
        {
            'epoch': num_epochs,
            'learning_rate': learning_rate,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        },
        f'{model_name}'
    )


# %%
def get_combined_data_df(dataset_container: MachineDatasetContainer, save_to_file: bool = True, is_training: bool = True) -> pd.DataFrame:
    print('create combined df')
    model.eval()
    prediction_list: List[pd.DataFrame] = list()
    actual_data_list: List[pd.DataFrame] = list()
    plan_data_list: List[pd.DataFrame] = list()
    
    def add_prediction(dataset: MachineDataset, dst: List[pd.DataFrame]):
        X_df = dataset.X.to(device)
        pred = model(X_df)
        
        if torch.has_cuda:
            pred = pred.cpu().detach().numpy()
        else: 
            pred = pred.data.numpy()
            
        label_columns = dataset.label_columns
        
        prediction_df = pd.DataFrame(pred, columns=label_columns)
        prediction_df = dataset.y_scaler.inverse_normalization_df(prediction_df)
        prediction_list.append(prediction_df)
        
    def add_actual_data(dataset: MachineDataset, dst: List[pd.DataFrame]):
        label_columns = dataset.label_columns
        actual_data = dataset.y.data.numpy()
        actual_data_df = pd.DataFrame(actual_data, columns=label_columns)
        actual_data_df = dataset.y_scaler.inverse_normalization_df(actual_data_df)
        actual_data_list.append(actual_data_df)
    
    def add_plan_data(dataset: MachineDataset, dst: List[pd.DataFrame]):
        plan_df = dataset.X.cpu()
        plan_df = dataset.X_scaler.convert_tensor_to_df(plan_df)
        plan_df = dataset.X_scaler.inverse_standardize_df(plan_df)
        plan_df = plan_df[['plan_cpu', 'plan_mem']]
        plan_data_list.append(plan_df)
        
    def get_rename_columns() -> dict:
        return {
            'cpu_usage_x': 'actual cpu usage', 
            'cpu_usage_y': 'predicted cpu usage', 
            'plan_cpu': 'allocated cpu',
            'avg_mem_x': 'actual mem usage',
            'avg_mem_y': 'predicted mem usage',
            'plan_mem': 'allocated mem'
            }
        
    def combine_dataframes(pred_df, actual_data_df, plan_df) -> pd.DataFrame:
        combined_df = pd.merge(actual_data_df, pred_df, left_index=True, right_index=True)
        combined_df[['plan_cpu', 'plan_mem']] = plan_df
        combined_df = combined_df.rename(columns=get_rename_columns())
        
        return combined_df
    
    for ds in dataset_container.dataset_list:
        add_prediction(ds, prediction_list)
        add_actual_data(ds, actual_data_list)
        add_plan_data(ds, plan_data_list)
        
    prediction_df = pd.concat(prediction_list, ignore_index=True)
    actual_data_df = pd.concat(actual_data_list, ignore_index=True)
    plan_df = pd.concat(plan_data_list, ignore_index=True)
    
    combined_df = combine_dataframes(prediction_df, actual_data_df, plan_df)

    if save_to_file and yaml_config['evaluation_path']['save_to_file']:
        prediction_path = 'training_prediction_path' if is_training else 'test_prediction_path'
        combined_df.to_csv(yaml_config['evaluation_path'][prediction_path])   
        
    del prediction_list, actual_data_list, plan_data_list
    
    return combined_df
        

# %%
train_combined_df = get_combined_data_df(dataset)

# %%
test_combined_df = get_combined_data_df(test_dataset, is_training=False)



