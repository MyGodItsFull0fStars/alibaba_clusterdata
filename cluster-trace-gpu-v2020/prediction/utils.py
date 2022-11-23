import math
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import nvidia_smi

def get_df(file: str, header=None, sample: bool = False, sample_number: int = 1000):
    if sample:
        df = pd.read_csv(file, header=None).sample(sample_number)
    else:
        df = pd.read_csv(file, header=None)

    df.columns = pd.read_csv("{}.header".format(
        file.split('.csv')[0])).columns if header is None else header
    return df


def get_device() -> torch.device:
    return torch.device(get_device_as_string())


def get_device_as_string() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_rmse(actual_values, predicted_values) -> float:
    '''returns the root mean squared error'''
    return math.sqrt(mean_squared_error(actual_values, predicted_values))


def get_mape(actual_values, predicted_values):
    '''returns the mean absolute percentage error'''
    return np.mean(np.abs(actual_values - predicted_values) / np.abs(actual_values) * 100)


def get_mae(actual_values, predicted_values) -> float:
    '''returns the mean absolute error'''
    return mean_absolute_error(actual_values, predicted_values)


def get_available_cuda_devices(free_mem_threshold: float = 0.90) -> List[str]:
    available_gpus: List[str] = []
    if get_device_as_string() == 'cpu':
        print('No CUDA device available')
    
    elif get_device_as_string() == 'cuda':
        nvidia_smi.nvmlInit()
        device_count = nvidia_smi.nvmlDeviceGetCount()
        
        for idx in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(idx)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            
            free_memory: float = info.free / info.total
            
            if free_memory >= free_mem_threshold:
                available_gpus.append(f'cuda:{idx}')
                
        nvidia_smi.nvmlShutdown()
    return available_gpus
                