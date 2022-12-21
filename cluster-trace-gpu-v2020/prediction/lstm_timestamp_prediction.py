# %%
print('import libraries')
from lstm_models import LSTM, UtilizationLSTM
from gpu_dataloader import ForecastDataset, UtilizationDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from sklearn.metrics import mean_absolute_error

from typing import List

# plotting the data
import matplotlib.pyplot as plt
# used for the dataframes
import pandas as pd
from tqdm import tqdm

from utils import get_device

import numpy as np
from utils import get_device_as_string, get_device, get_rmse, get_mae

import yaml
from yaml.loader import SafeLoader

# %%
# Open the file and load the file
with open('./model_configs/tasks_vs_no_tasks/utilization_no_tasks.yaml') as f:
    yaml_config = yaml.load(f, Loader=SafeLoader)

# %%
batch_size: int = yaml_config['dataset']['batch_size']
small_df: bool = yaml_config['dataset']['small_df']
include_tasks: bool = yaml_config['dataset']['include_tasks']

# %%
print('load datasets')
dataset = UtilizationDataset(is_training=True, small_df=small_df, include_tasks=include_tasks)
test_set = UtilizationDataset(is_training=False, small_df=small_df, include_tasks=include_tasks)

# %%
print('init model parameters')
num_epochs: int = yaml_config['model']['num_epochs']
learning_rate: float = yaml_config['model']['learning_rate']

# number of features
input_size: int = dataset.X.shape[2]
# number of features in hidden state
hidden_size: int = yaml_config['model']['hidden_size']
# number of stacked lstm layers
num_layers: int = yaml_config['model']['num_layers']
# number of output classes
num_classes: int = dataset.y.shape[1]

print(f'input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, num_classes: {num_classes}')
device = get_device()

INCLUDE_WANDB: bool = False

# %%
if INCLUDE_WANDB == True:
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
print('init model')
model = LSTM(num_classes, input_size, hidden_size, num_layers)
# model = UtilizationLSTM(num_classes, input_size, hidden_size, num_layers)
model.train()

# log gradients and model parameters
if INCLUDE_WANDB:
    wandb.watch(model)

# %%
# mean square error for regression
print('init loss, optimizer and scheduler')
criterion = nn.MSELoss()
# criterion = RMSELoss()
criterion = criterion.to(device)
# optimizer function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler_config = yaml_config['model']['scheduler']
patience = scheduler_config['patience']
factor = scheduler_config['factor']
min_lr = scheduler_config['min_lr']
eps = scheduler_config['eps']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr, eps=eps)


# %%
def log_training_metrics(predictions, labels, loss):
    # logging to wandb
    if get_device_as_string() == 'cuda' or get_device_as_string() == 'mps':
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
def reorder_dataset(dataset: ForecastDataset, batch_size: int):
    batch_order = np.array([batch for batch in range(0, len(dataset), batch_size)], dtype=np.int32)
    batch_order = np.random.permutation(batch_order)

    dataset_order = np.empty(shape=[0, len(dataset.X)], dtype=np.int32)

    for batch in batch_order:
        if batch >= len(dataset) - (batch_size - 1):
            continue
        filled_batch_order = np.arange(batch, batch + batch_size, dtype=np.int32)
        dataset_order = np.append(dataset_order, filled_batch_order)
        
    dataset.X = dataset.X[dataset_order]
    dataset.y = dataset.y[dataset_order]

# %%
modulo_switch = num_epochs // 10
print('init train dataloader')
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)

loss_val = None
loss_progression: list = []

# %%
if torch.has_cuda:
    torch.cuda.empty_cache()

# %%
def training_loop(train_loader: DataLoader) -> float:
    predictions, labels, loss = 0, 0, 0
    for _, (inputs, labels) in enumerate(train_loader):
        # send input and label to device
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.shape, labels.shape)
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
        val_pred = model(test_set.X.to(device))
        val_loss = criterion(val_pred, test_set.y.to(device))
        scheduler.step(val_loss)

# %%
print('start training loop')
for epoch in (pbar := tqdm(range(0, num_epochs), desc=f'Training Loop (0) -- Loss: {loss_val}')):

    # if epoch % modulo_switch == modulo_switch - 1:
    #     reorder_dataset(dataset, batch_size // 2)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    loss_val = training_loop(train_loader)
    loss_progression.append(loss_val)
    pbar.set_description(f'Training Loop ({epoch + 1}) -- Loss: {loss_val:.5f}')
    
    validation_loop()

# %%
loss_df = pd.DataFrame(data=loss_progression)
if yaml_config['evaluation_path']['save_to_file'] == True:
    loss_df.to_csv(yaml_config['evaluation_path']['loss_progression'])

# %%
import time

current_time = time.ctime()
current_time

# %% [markdown]
# ## Save the Model to Disk

# %%
model.eval()
print('save model')
if yaml_config['model']['save_model']:
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
def get_combined_data_df(data_set: UtilizationDataset, save_to_file: bool = True, is_training: bool = True) -> pd.DataFrame:
    model.eval()
        
    X_df = data_set.X.to(device)
    label_columns: List[str] = dataset.get_label_columns()
    
    def get_prediction_df() -> pd.DataFrame:
        prediction = model(X_df)
        if torch.has_cuda:
            prediction = prediction.cpu().detach().numpy()
        else:
            prediction = prediction.data.numpy()
            
        prediction_df = pd.DataFrame(prediction, columns=label_columns)
        prediction_df = data_set.y_scaler.inverse_normalization_df(prediction_df)
        
        return prediction_df
    
    def get_actual_data_df() -> pd.DataFrame:
        actual_data = data_set.y.data.numpy()
        actual_data_df = pd.DataFrame(actual_data, columns=label_columns)
        actual_data_df = data_set.y_scaler.inverse_normalization_df(actual_data_df)
        
        return actual_data_df
    
    def get_plan_data_df() -> pd.DataFrame:
        plan = data_set.X.cpu()
        plan_df = dataset.X_scaler.convert_tensor_to_df(plan)
        plan_df = dataset.X_scaler.inverse_standardize_df(plan_df)
        plan_df = plan_df[['plan_cpu', 'plan_mem']]
        return plan_df
    
    prediction_df = get_prediction_df()
    actual_data_df = get_actual_data_df()
    plan_df = get_plan_data_df()
    
    rename_columns_dict: dict = {
        'cpu_usage_x': 'actual cpu usage',
        'cpu_usage_y': 'predicted cpu usage',
        'plan_cpu': 'allocated cpu',
        'avg_mem_x': 'actual mem usage',
        'avg_mem_y': 'predicted mem usage',
        'plan_mem': 'allocated mem'
    }
    
    combined_df = pd.merge(actual_data_df, prediction_df, left_index=True, right_index=True)
    # combined_df.rename()
    combined_df[['plan_cpu', 'plan_mem']] = plan_df

    combined_df = combined_df.rename(columns=rename_columns_dict)
    
    if yaml_config['evaluation_path']['save_to_file'] and save_to_file:
        prediction_path: str = 'training_prediction_path' if is_training else 'test_prediction_path'
        combined_df.to_csv(yaml_config['evaluation_path'][prediction_path])
        
    return combined_df

# %%
print('save combined dfs')
combined_train_df = get_combined_data_df(dataset, is_training=True)

# %%
combined_test_df = get_combined_data_df(test_set, is_training=False)