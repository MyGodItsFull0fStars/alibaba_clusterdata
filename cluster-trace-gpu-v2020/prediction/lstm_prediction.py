# %%
from lstm_models import LSTM, UtilizationLSTM
from gpu_dataloader import ForecastDataset, UtilizationDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from sklearn.metrics import mean_absolute_error

# plotting the data
import matplotlib.pyplot as plt
# used for the dataframes
import pandas as pd
from tqdm import tqdm

from utils import get_device

import numpy as np

from utils import get_device_as_string, get_device, get_rmse, get_mae

# %%
import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open('./model_configs/tasks_vs_no_tasks/utilization_no_tasks.yaml') as f:
    yaml_config = yaml.load(f, Loader=SafeLoader)
    # print(yaml_config)

# %%
batch_size: int = yaml_config['dataset']['batch_size']
small_df: bool = yaml_config['dataset']['small_df']
include_tasks: bool = yaml_config['dataset']['include_tasks']

print('init dataset')
# %%
dataset = UtilizationDataset(small_df=small_df, include_tasks=include_tasks)
test_set = UtilizationDataset(is_training=False, small_df=small_df, include_tasks=include_tasks)

print('init hyperparameters')
# %%
num_epochs: int = yaml_config['model']['num_epochs']
learning_rate: float = yaml_config['model']['learning_rate']

# number of features
input_size: int = dataset.X.shape[2]
# number of features in hidden state
# hidden_size: int = dataset.X.shape[2] * 1000
hidden_size: int = yaml_config['model']['hidden_size']
# number of stacked lstm layers
num_layers: int = yaml_config['model']['num_layers']
# number of output classes
num_classes: int = dataset.y.shape[1]
seq_length: int = dataset.X.shape[1]

INCLUDE_WANDB: bool = yaml_config['logging']['enable_wandb']

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
# lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length, bidirectional=bidirectional)
print('init lstm')
lstm = UtilizationLSTM(num_classes, input_size, hidden_size, num_layers)
lstm.train()

device = lstm.device

# log gradients and model parameters
if INCLUDE_WANDB:
    wandb.watch(lstm)

# %%
# mean square error for regression
# nn.
criterion = nn.MSELoss()
criterion = criterion.to(device)
# criterion = RMSELoss()
# criterion = criterion.to(device)
# optimizer function
optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)

scheduler_config = yaml_config['model']['scheduler']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=scheduler_config['patience'], factor=scheduler_config['factor'], min_lr=scheduler_config['min_lr'], eps=scheduler_config['eps'])


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
    # print(log_dict)

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
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)

loss_val = None
loss_progression: list = []

# %%
if torch.has_cuda:
    torch.cuda.empty_cache()

print('start training')
# %%
for epoch in (pbar := tqdm(range(0, num_epochs), desc=f'Training Loop (0) -- Loss: {loss_val}')):

    # if epoch % modulo_switch == modulo_switch - 1:
    #     reorder_dataset(dataset, batch_size // 2)
    #     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    for _, (inputs, labels) in enumerate(train_loader):
        # send input and label to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward input to model
        predictions = lstm(inputs).to(device)

        optimizer.zero_grad()
        loss = criterion(predictions, labels)
        # backward propagation
        loss.backward()
        # update weights
        optimizer.step()
        
        
    if INCLUDE_WANDB:
        log_training_metrics(predictions, labels, loss)
    loss_val = loss.item()
    loss_progression.append(loss_val)
    pbar.set_description(f'Training Loop ({epoch + 1}) -- Loss: {loss_val:.5f}')
    
    with torch.no_grad():
        val_pred = lstm(test_set.X.to(device))
        val_loss = criterion(val_pred, test_set.y.to(device))
        scheduler.step(val_loss)
        


# %%
# loss_df = pd.DataFrame(data=loss_progression)
# loss_df.plot.line()

# %%
import time

current_time = time.ctime()

# %% [markdown]
# ## Save the Model to Disk

# %%
lstm.eval()

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
            'seq_length': seq_length,
            'model_state_dict': lstm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        },
        f'{model_name}'
    )


# %%
X_df = dataset.X
X_df = X_df.to(device)
# forward pass
prediction = lstm(X_df)
# prediction = prediction.reshape(1, -1)
if torch.cuda.is_available():
    prediction = prediction.cpu().detach().numpy()
else:
    prediction = prediction.data.numpy()

actual_data = dataset.y.data.numpy()
label_columns = dataset._get_label_columns()

# create dataframes
prediction_df = pd.DataFrame(prediction, columns=label_columns)
actual_data_df = pd.DataFrame(actual_data, columns=label_columns)

# reverse transformation
prediction_df = dataset.y_scaler.inverse_normalization_df(prediction_df)
actual_data_df = dataset.y_scaler.inverse_normalization_df(actual_data_df)


# %%
plan = X_df.cpu()
plan_df = dataset.X_scaler.convert_tensor_to_df(plan)
plan_df = dataset.X_scaler.inverse_standardize_df(plan_df)
plan_df = plan_df[['plan_cpu', 'plan_mem']]

# %%

rename_columns_dict: dict = {
    'cpu_usage_x': 'actual cpu usage', 
    'cpu_usage_y': 'predicted cpu usage', 
    'plan_cpu': 'allocated cpu',
    'avg_mem_x': 'actual mem usage',
    'avg_mem_y': 'predicted mem usage',
    'plan_mem': 'allocated mem'
    }

# %% [markdown]
# ## Calculate Root Mean Squared Error
# 
# Calculating the RMSE for the overall prediction of the (training) dataset.

# %%
rmse_key: str = 'Root Mean Squared Error (Overall - Training)'
rmse_result = get_rmse(actual_data_df[:], prediction_df[:])
print(f'Test Score: {rmse_result:.2f} RMSE')
if INCLUDE_WANDB:
    wandb.summary[rmse_key] = rmse_result

# %% [markdown]
# ## Calculate Mean Absolute Error
# 
# Calcutlate the MAE for the overall prediction of the (training) dataset.

# %%
mae_key: str = 'Mean Absolute Error (Overall - Training)'
mae_result = mean_absolute_error(actual_data_df[:], prediction_df[:])
print(f'Test Score: {mae_result} MAE')

if INCLUDE_WANDB:
    wandb.summary[mae_key] = mae_result

# %%
combined_df = pd.merge(actual_data_df, prediction_df, left_index=True, right_index=True)
# combined_df.rename()
combined_df[['plan_cpu', 'plan_mem']] = plan_df

combined_df = combined_df.rename(columns=rename_columns_dict)

combined_df['rmse'] = rmse_result
combined_df['mae'] = mae_result

if yaml_config['evaluation_path']['save_to_file']:
    combined_df.to_csv(yaml_config['evaluation_path']['training_prediction_path'])

# %%
def plot_column(actual_values=actual_data_df, predicted_values=prediction_df, column_number: int = 0, rmse_threshold: float = 0.30, is_training: bool = True):

    if len(label_columns) <= column_number:
        print('Out of Prediction Bounds')
        return

    plt.figure(figsize=(25, 15))  # plotting
    plt.rcParams.update({'font.size': 22})

    column = label_columns[column_number]
    pred_column = f"pred_{column}_{'training' if is_training else 'test'}"

    rmse = get_rmse(actual_values[column], predicted_values[column])
    mae = mean_absolute_error(actual_values[column], predicted_values[column])

    predicted_color = 'green' if rmse < rmse_threshold else 'orange'

    plt.plot(actual_values[column], label=column, color='black')  # actual plot
    plt.plot(predicted_values[column], label='pred_' +
             column, color=predicted_color)  # predicted plot

    plt.title('Time-Series Prediction')
    plt.plot([], [], ' ', label=f'RMSE: {rmse}')
    plt.plot([], [], ' ', label=f'MAE: {mae}')
    plt.legend()
    plt.ylabel('timeline', fontsize=25)
    
    if INCLUDE_WANDB:
        wandb.log({pred_column: wandb.Image(plt)})
        wandb.summary[f'Root Mean Squared Error ({column})'] = rmse
        wandb.summary[f'Mean Absolute Error ({column})'] = mae
        
    plt.show()


# %% [markdown]
# ## See Predictions on Training Dataset

# %%
# for idx in range(0, len(label_columns)):
#     plot_column(actual_values=actual_data_df, predicted_values=prediction_df, column_number=idx)

# %% [markdown]
# ## Test Set Analysis
# 
# Below, the test set will be loaded and the model evaluated with it to see the actual performance.

# %%
lstm.eval()

X_df = test_set.X
X_df = X_df.to(device)
# forward pass
prediction = lstm(X_df)
if torch.cuda.is_available():
    prediction = prediction.cpu().data.numpy()
else:
    prediction = prediction.data.numpy()

actual_data = test_set.y.data.numpy()

label_columns = test_set._get_label_columns()

# create dataframes
prediction_df = pd.DataFrame(prediction, columns=label_columns)
actual_data_df = pd.DataFrame(actual_data, columns=label_columns)

# reverse transformation
prediction_df = dataset.y_scaler.inverse_normalization_df(prediction_df)
actual_data_df = dataset.y_scaler.inverse_normalization_df(actual_data_df)


# %%
plan = X_df.cpu()
plan_test_df = test_set.X_scaler.convert_tensor_to_df(plan)
plan_test_df = test_set.X_scaler.inverse_standardize_df(plan_test_df)
plan_test_df = plan_df[['plan_cpu', 'plan_mem']]

# %%
rmse_key: str = 'Root Mean Squared Error (Overall - Test)'
rmse_result = get_rmse(actual_data_df[:], prediction_df[:])
print(f'Test Score: {rmse_result:.2f} RMSE')

if INCLUDE_WANDB:
    wandb.summary[rmse_key] = rmse_result

# %%
mae_key: str = 'Mean Absolute Error (Overall - Test)'
mae_result = mean_absolute_error(actual_data_df[:], prediction_df[:])
print(f'Test Score: {mae_result} MAE')
if INCLUDE_WANDB:
    wandb.summary[mae_key] = mae_result

# %%
combined_test_df = pd.merge(actual_data_df, prediction_df, left_index=True, right_index=True)
# combined_df.rename()
combined_test_df[['plan_cpu', 'plan_mem']] = plan_test_df 

combined_test_df = combined_test_df.rename(columns=rename_columns_dict)

combined_test_df['rmse'] = rmse_result
combined_test_df['mae'] = mae_result

if yaml_config['evaluation_path']['save_to_file']:
    combined_test_df.to_csv(yaml_config['evaluation_path']['test_prediction_path'])


