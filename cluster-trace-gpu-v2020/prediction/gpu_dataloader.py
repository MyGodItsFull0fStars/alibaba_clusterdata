from typing import List, Tuple
import torch
from torch import Tensor, is_distributed, std
from torch.utils.data import Dataset

from time import perf_counter
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from dataframe_scaler import DataFrameScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler

TRAINING_DATAPATH: str = 'training_df.csv'
TEST_DATAPATH: str = 'test_df.csv'

MEAN_KEY: str = 'mean'
STD_DEV_KEY: str = 'std'

# adding this line because standardizing the dataframe is printing a false positive error message
# source: https://stackoverflow.com/a/42190404
pd.options.mode.chained_assignment = None  # type: ignore


class GPUDataset(Dataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        batch_size: int = 1000,
        future_step: int = 10,
        small_df: bool = False
    ) -> None:
        super().__init__()

        self.data_path: str = TRAINING_DATAPATH if is_training else TEST_DATAPATH
        self.data_path = self.__prepare_data_path(self.data_path)
        self.data_index = self.__prepare_data_index(data_index)

        self.batch_size: int = batch_size
        self.small_df: bool = small_df

        self.X_scaler: DataFrameScaler = DataFrameScaler()
        self.y_scaler: DataFrameScaler = DataFrameScaler()

        self.X: Tensor = torch.Tensor()
        self.y: Tensor = torch.Tensor()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        if 0 <= index < self.X.size(0):
            return self.X[index], self.y[index]

    def _prepare_data_tensors(self) -> Tuple[Tensor, Tensor]:
        return torch.empty(0), torch.empty(0)

    def _read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.set_index(self.data_index, inplace=True)
        df.dropna(inplace=True)

        return df

    def _get_feature_label_tensors(self) -> Tuple[Tensor, Tensor]:
        return torch.empty(0), torch.empty(0)

    def _get_feature_columns(self) -> List[str]:
        return []

    def _get_label_columns(self) -> List[str]:
        return []

    def _get_job_columns(self) -> List[str]:
        return [
            'BladeMain',
            'JupyterTask', 'OpenmpiWorker', 'OssToVolumeWorker', 'PyTorchWorker',
            'TVMTuneMain', 'chief', 'evaluator', 'ps', 'tensorflow', 'worker',
            'xComputeWorker'
        ]

    def _get_cpu_utilization_columns(self) -> List[str]:
        return ['cpu_usage']

    def _get_mem_utilization_columns(self) -> List[str]:
        return self.get_avg_mem_utilization_column() + self.get_max_mem_utilization_column()
    
    def get_avg_mem_utilization_column(self) -> List[str]:
        return ['avg_mem']
    
    def get_max_mem_utilization_column(self) -> List[str]:
        return ['max_mem']

    def _get_gpu_utilization_columns(self) -> List[str]:
        return ['gpu_wrk_util',
                'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']

    def _get_runtime_column(self) -> List[str]:
        return ['runtime']

    def _get_plan_cpu_columns(self) -> List[str]:
        return ['plan_cpu']

    def _get_plan_mem_columns(self) -> List[str]:
        return ['plan_mem']

    def _get_plan_gpu_columns(self) -> List[str]:
        return ['plan_gpu']

    def _get_plan_utilization_columns(self) -> List[str]:
        return self._get_plan_cpu_columns() + self._get_plan_gpu_columns() + self._get_plan_mem_columns()

    def get_cap_cpu_columns(self) -> List[str]:
        return ['cap_cpu']

    def get_cap_gpu_columns(self) -> List[str]:
        return ['cap_gpu']

    def get_cap_mem_columns(self) -> List[str]:
        return ['cap_mem']

    def _get_cap_utilization_columns(self) -> List[str]:
        return self.get_cap_cpu_columns() + self.get_cap_gpu_columns() + self.get_cap_mem_columns()

    def __prepare_data_path(self, data_path: str) -> str:
        if data_path is None or len(data_path) == 0:
            data_path = 'machine_sorted_df.csv'
        return data_path

    def __prepare_data_index(self, data_index) -> str:
        if data_index is None or len(data_index) == 0:
            data_index = 'start_date'
        return data_index

    def _resize_df(self, df: pd.DataFrame, split_index: int = 4000) -> pd.DataFrame:
        if self.small_df:
            return df.iloc[:split_index]
        else:
            return df.iloc[:1954000]

    def _transform_dfs_to_tensors(self, X_df, y_df) -> Tuple[torch.Tensor, torch.Tensor]:
        X_df, y_df = X_df.to_numpy(), y_df.to_numpy()

        # Convert to Tensors
        X_df = torch.Tensor(X_df)
        y_df = torch.Tensor(y_df)

        # Reshape Feature Tensor
        X_df = torch.reshape(X_df, (X_df.shape[0], 1, X_df.shape[1]))

        return X_df, y_df

class MachineSplitDataset():
    
    def __init__(
        self,
        is_training: bool = True,
        small_df: bool = False,
        include_tasks: bool = False
        ) -> None:
        
        self.is_training: bool = is_training
        self.small_df: bool = small_df
        self.include_tasks: bool = include_tasks
        
        self.start_index_array = np.empty((0, 0))
        self.end_index_array = np.empty((0, 0))
        self.init_index_arrays()
        
        self.dataset_list: List[Dataset] = self.init_datasets()
        
        
    def init_datasets(self) -> List[Dataset]:
        machine_df = self.read_data_csv()
        
        def get_machine_list() -> list:
            machine_list = list()
            if self.small_df is False:
                for idx in range(len(self.start_index_array)):
                    machine = machine_df.iloc[:, self.start_index_array[idx]:self.end_index_array[idx]]
                    machine_list.append(machine)
                    
            else:
                for idx in range(5):
                    machine = machine_df.iloc[:, self.start_index_array[idx]:self.end_index_array[idx]]
                    machine_list.append(machine)
            return machine_list
        
        machine_list = get_machine_list()
        
        return list()
    
    def init_index_arrays(self):
        index_df = self.read_index_csv()
        
        if self.is_training:
            self.start_index_array = index_df['train_start'].values
            self.end_index_array = index_df['train_end'].values
        else: # test set
            self.start_index_array = index_df['test_start'].values
            self.end_index_array = index_df['test_end'].values
            
        
    def read_data_csv(self, csv_path: str = 'df_machine_sorted.csv') -> pd.DataFrame:
        return pd.read_csv(csv_path)
    
    def read_index_csv(self, csv_path: str = 'machine_indices.csv') -> pd.DataFrame:
        return pd.read_csv(csv_path, index_col=0)

class UtilizationDataset(GPUDataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        future_step: int = 10,
        small_df: bool = False,
        include_tasks: bool = False
    ) -> None:
        super(UtilizationDataset, self).__init__(is_training, data_index,
                                                 1000, future_step, small_df)  # type: ignore
        self.include_tasks = include_tasks

        self.X, self.y = self._prepare_data_tensors()

    def _prepare_data_tensors(self) -> Tuple[Tensor, Tensor]:
        df = self._read_csv()
        df = self._resize_df(df)

        return self._init_data_tensors(df=df)

    def _init_data_tensors(self, df: pd.DataFrame) -> Tuple[Tensor, Tensor]:

        X_df = df[self._get_feature_columns()]
        X_df[self.get_cap_cpu_columns()] = X_df[self.get_cap_cpu_columns()]  
        
        y_df = df[self._get_label_columns()]

        self.X_scaler = DataFrameScaler(X_df, self._get_job_columns())
        self.y_scaler = DataFrameScaler(y_df, self._get_job_columns())

        X_df = self.X_scaler.standardize_df(X_df)
        y_df = self.y_scaler.normalize_df(y_df)
        
        # X_df.to_csv(f'./datasets/standardized_feature_df.csv')
        # y_df.to_csv(f'./datasets/normalized_label_df.csv')
        # self.X_scaler.std_dev_df.to_csv(f'./datasets/standard_feature_df.csv')
        # self.y_scaler.norm_dev_df.to_csv(f'./datasets/norm_label_df.csv')
        
        X_tens, y_tens = self._transform_dfs_to_tensors(X_df, y_df)

        return X_tens, y_tens

    def _get_feature_columns(self) -> List[str]:
        if self.include_tasks == True:
            return self._get_plan_cpu_columns() + self._get_plan_mem_columns() + self.get_cap_cpu_columns() + self.get_cap_mem_columns() + self._get_job_columns()
        return self._get_plan_cpu_columns() + self.get_cap_cpu_columns() + self._get_plan_mem_columns() + self.get_cap_mem_columns()

    def _get_label_columns(self) -> List[str]:
        return self._get_cpu_utilization_columns() + self.get_avg_mem_utilization_column()
        # return self._get_cpu_utilization_columns() + self._get_mem_utilization_columns() + self._get_runtime_column()


class ForecastDataset(GPUDataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        batch_size: int = 1000,
        future_step: int = 10,
        small_df: bool = False
    ) -> None:
        super(ForecastDataset, self).__init__(is_training,
                                              data_index, batch_size, future_step, small_df)

        self.X, self.y = self._prepare_data_tensors()
        self.X_orig, self.y_orig = torch.clone(self.X), torch.clone(self.y)
        self.num_samples: int = self.X.shape[0]

    def _prepare_data_tensors(self) -> Tuple[Tensor, Tensor]:
        df = self._read_csv()
        df = self._resize_df(df)

        return self.__append_to_feature_and_label_set(df)

    def _get_feature_columns(self) -> List[str]:
        return self._get_cpu_utilization_columns() + self._get_mem_utilization_columns() + self._get_gpu_utilization_columns() + self._get_runtime_column() + self._get_job_columns()

    def _get_label_columns(self) -> List[str]:
        # return ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
        #                      'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime']
        return self._get_runtime_column()

    def _get_feature_label_tensors(self) -> Tuple[Tensor, Tensor]:
        return self.X, self.y

    def _get_all_machines(self, df: pd.DataFrame) -> np.ndarray:
        return df.machine.unique()

    def __append_to_feature_and_label_set(self, df: pd.DataFrame):

        X_df = pd.DataFrame()
        y_df = pd.DataFrame()

        for step in range(0, len(df) // self.batch_size, 2):

            feature_start_index = step * self.batch_size
            feature_end_index = feature_start_index + self.batch_size

            label_start_index = feature_end_index
            label_end_index = feature_end_index + self.batch_size

            if label_end_index > len(df):
                break

            X_df = pd.concat(
                [X_df, df.iloc[feature_start_index:feature_end_index]])
            y_df = pd.concat(
                [y_df, df.iloc[label_start_index:label_end_index]])

        X_df, y_df = self.__filter_columns(X_df, y_df)
        
        X_df = self.X_scaler.fit_transform_std(X_df)
        y_df = self.y_scaler.fit_transform_norm(y_df)

        return self._transform_dfs_to_tensors(X_df, y_df)

    def __filter_columns(self, X_df, y_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = X_df[self._get_feature_columns()]
        y_df = y_df[self._get_label_columns()]

        return X_df, y_df


if __name__ == '__main__':

    # test_dataset = GPUDataset(small_df=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)

    # test_dataset = UtilizationDataset(small_df=True, include_tasks=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    
    test = MachineSplitDataset()
    test.read_data_csv()
    test.read_index_csv()
    

    # test_dataset = ForecastDataset(small_df=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    # dataset = GPUDataset(is_training=True, small_df=True)
    # print(dataset.X.shape, dataset.y.shape)
    # print(dataset.X)
