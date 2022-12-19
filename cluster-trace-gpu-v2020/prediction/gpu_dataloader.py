from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset

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

def transform_dfs_to_tensors(X_df, y_df) -> Tuple[torch.Tensor, torch.Tensor]:
        X_df, y_df = X_df.to_numpy(), y_df.to_numpy()

        # Convert to Tensors
        X_df = torch.Tensor(X_df)
        y_df = torch.Tensor(y_df)

        # Reshape Feature Tensor
        X_df = torch.reshape(X_df, (X_df.shape[0], 1, X_df.shape[1]))

        return X_df, y_df

class DatasetColumns(object):
    
    @staticmethod
    def _get_job_columns() -> List[str]:
        return [
            'BladeMain',
            'JupyterTask', 'OpenmpiWorker', 'OssToVolumeWorker', 'PyTorchWorker',
            'TVMTuneMain', 'chief', 'evaluator', 'ps', 'tensorflow', 'worker',
            'xComputeWorker'
        ]

    @staticmethod
    def _get_cpu_utilization_columns() -> List[str]:
        return ['cpu_usage']

    @staticmethod
    def _get_mem_utilization_columns() -> List[str]:
        return DatasetColumns.get_avg_mem_utilization_column() + DatasetColumns.get_max_mem_utilization_column()
    
    @staticmethod
    def get_avg_mem_utilization_column() -> List[str]:
        return ['avg_mem']
    
    @staticmethod
    def get_max_mem_utilization_column() -> List[str]:
        return ['max_mem']

    @staticmethod
    def _get_gpu_utilization_columns() -> List[str]:
        return ['gpu_wrk_util',
                'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']

    @staticmethod
    def _get_runtime_column() -> List[str]:
        return ['runtime']

    @staticmethod
    def _get_plan_cpu_columns() -> List[str]:
        return ['plan_cpu']

    @staticmethod
    def _get_plan_mem_columns() -> List[str]:
        return ['plan_mem']

    @staticmethod
    def _get_plan_gpu_columns() -> List[str]:
        return ['plan_gpu']

    @staticmethod
    def _get_plan_utilization_columns() -> List[str]:
        return DatasetColumns._get_plan_cpu_columns() + DatasetColumns._get_plan_gpu_columns() + DatasetColumns._get_plan_mem_columns()

    @staticmethod
    def get_cap_cpu_columns() -> List[str]:
        return ['cap_cpu']

    @staticmethod
    def get_cap_gpu_columns() -> List[str]:
        return ['cap_gpu']

    @staticmethod
    def get_cap_mem_columns() -> List[str]:
        return ['cap_mem']

    @staticmethod
    def _get_cap_utilization_columns() -> List[str]:
        return DatasetColumns.get_cap_cpu_columns() + DatasetColumns.get_cap_gpu_columns() + DatasetColumns.get_cap_mem_columns()


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

    def get_feature_columns(self) -> List[str]:
        return []

    def get_label_columns(self) -> List[str]:
        return []

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
    
class MachineDataset(Dataset):
    
    def __init__(self, machine: pd.DataFrame, feature_columns: List[str], label_columns: List[str]) -> None:
        super(MachineDataset, self).__init__()
        
        self.X: Tensor = torch.empty(0)
        self.y: Tensor = torch.empty(0)
        self.X_scaler: DataFrameScaler = DataFrameScaler()
        self.y_scaler: DataFrameScaler = DataFrameScaler()
        
        self.feature_columns: List[str] = feature_columns
        self.label_columns: List[str] = label_columns
        
        self.prepare_tensors(machine)
        
        
    def prepare_tensors(self, machine: pd.DataFrame) -> None:
        X_df = machine[self.feature_columns]
        y_df = machine[self.label_columns]
        
        self.X_scaler = DataFrameScaler(X_df, filter_columns=DatasetColumns._get_job_columns())
        self.y_scaler = DataFrameScaler(y_df, filter_columns=DatasetColumns._get_job_columns())
        
        X_df = self.X_scaler.standardize_df(X_df)
        y_df = self.y_scaler.normalize_df(y_df)
        
        self.X, self.y = transform_dfs_to_tensors(X_df, y_df)
        
        del X_df, y_df
        
        
    def get_feature_shape(self) -> torch.Size:
        return self.X.shape
        
    def get_model_input_size(self) -> int:
        input_size = self.X.shape[2]
        assert input_size is not None and input_size > 0
        return input_size
        
        
    def get_model_num_classes(self) -> int:
        num_classes = self.y.shape[1]
        assert num_classes is not None and num_classes > 0
        return num_classes
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index):
        if 0 <= index < self.X.size(0):
            return self.X[index], self.y[index]

class MachineDatasetContainer():
    
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
        
        self.dataset_list: List[MachineDataset] = list()
        
        self.init_index_arrays()
        self.init_machine_dataset()
        
        
    def init_machine_dataset(self):
        machine_df = self.read_csv('df_machine_sorted.csv')
        
        def get_machine_list() -> list:
            machine_list = list()
            idx_range = 3 if self.small_df else len(self.start_index_array)

            for idx in range(idx_range):
                start = self.start_index_array[idx]
                end = self.end_index_array[idx]
                
                machine = machine_df.iloc[start:end]
                machine_list.append(machine)
                    
            return machine_list
        
        def get_machine_dataset(machine_list: List[pd.DataFrame]) -> List[MachineDataset]:
            dataset_list: List[MachineDataset] = list()
            for m in machine_list:
                m_ds = MachineDataset(m, self.get_feature_columns(), self.get_label_columns())
                dataset_list.append(m_ds)
                
            return dataset_list
        
        machine_list = get_machine_list()
        assert machine_list is not None and len(machine_list) > 0        
        
        self.dataset_list = get_machine_dataset(machine_list)
        
        del machine_df, machine_list
        
        
    def init_index_arrays(self):
        index_df = self.read_csv('machine_indices.csv')
        
        index_prefix = 'train' if self.is_training else 'test'
        self.start_index_array = index_df[f'{index_prefix}_start'].values
        self.end_index_array = index_df[f'{index_prefix}_end'].values
            
        del index_df
        
    def get_feature_columns(self) -> List[str]:
        if self.include_tasks == True:
            return DatasetColumns._get_plan_cpu_columns() + DatasetColumns._get_plan_mem_columns() + DatasetColumns.get_cap_cpu_columns() + DatasetColumns.get_cap_mem_columns() + DatasetColumns._get_job_columns()
        return DatasetColumns._get_plan_cpu_columns() + DatasetColumns.get_cap_cpu_columns() + DatasetColumns._get_plan_mem_columns() + DatasetColumns.get_cap_mem_columns()

    def get_label_columns(self) -> List[str]:
        return DatasetColumns._get_cpu_utilization_columns() + DatasetColumns.get_avg_mem_utilization_column()

    def read_csv(self, csv_path: str = '') -> pd.DataFrame:
        if len(csv_path) == 0:
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path, index_col=0)
        assert df is not None and len(df) > 0
        
        return df
    
    def get_model_input_size(self) -> int:
        return self.dataset_list[0].get_model_input_size()

    def get_model_num_classes(self) -> int:
        return self.dataset_list[0].get_model_num_classes()
    
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

        X_df = df[self.get_feature_columns()]
        # TODO does this make any sense?
        X_df[DatasetColumns.get_cap_cpu_columns()] = X_df[DatasetColumns.get_cap_cpu_columns()]  
        
        y_df = df[self.get_label_columns()]

        self.X_scaler = DataFrameScaler(X_df, DatasetColumns._get_job_columns())
        self.y_scaler = DataFrameScaler(y_df, DatasetColumns._get_job_columns())

        X_df = self.X_scaler.standardize_df(X_df)
        y_df = self.y_scaler.normalize_df(y_df)
        
        # X_df.to_csv(f'./datasets/standardized_feature_df.csv')
        # y_df.to_csv(f'./datasets/normalized_label_df.csv')
        # self.X_scaler.std_dev_df.to_csv(f'./datasets/standard_feature_df.csv')
        # self.y_scaler.norm_dev_df.to_csv(f'./datasets/norm_label_df.csv')
        
        X_tens, y_tens = self._transform_dfs_to_tensors(X_df, y_df)

        return X_tens, y_tens

    def get_feature_columns(self) -> List[str]:
        if self.include_tasks == True:
            return DatasetColumns._get_plan_cpu_columns() + DatasetColumns._get_plan_mem_columns() + DatasetColumns.get_cap_cpu_columns() + DatasetColumns.get_cap_mem_columns() + DatasetColumns._get_job_columns()
        return DatasetColumns._get_plan_cpu_columns() + DatasetColumns.get_cap_cpu_columns() + DatasetColumns._get_plan_mem_columns() + DatasetColumns.get_cap_mem_columns()

    def get_label_columns(self) -> List[str]:
        return DatasetColumns._get_cpu_utilization_columns() + DatasetColumns.get_avg_mem_utilization_column()
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

    def get_feature_columns(self) -> List[str]:
        return DatasetColumns._get_cpu_utilization_columns() + DatasetColumns._get_mem_utilization_columns() + DatasetColumns._get_gpu_utilization_columns() + DatasetColumns._get_runtime_column() + DatasetColumns._get_job_columns()

    def get_label_columns(self) -> List[str]:
        # return ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
        #                      'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime']
        return DatasetColumns._get_runtime_column()

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
        X_df = X_df[self.get_feature_columns()]
        y_df = y_df[self.get_label_columns()]

        return X_df, y_df


if __name__ == '__main__':

    # test_dataset = GPUDataset(small_df=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)

    # test_dataset = UtilizationDataset(small_df=True, include_tasks=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    
    cont = MachineDatasetContainer(small_df=True)
    first_machine = cont.dataset_list[0]
    # print(first_machine.X)
    print('fin')
    
    
    # test.read_data_csv()
    # test.read_index_csv()
    # single_machine_df = pd.read_csv('./single_machine.csv', index_col=0)    
    # feat = DatasetColumns._get_plan_cpu_columns() + DatasetColumns._get_plan_mem_columns() + DatasetColumns.get_cap_cpu_columns() + DatasetColumns.get_cap_mem_columns()
    # lab = DatasetColumns._get_cpu_utilization_columns() + DatasetColumns.get_avg_mem_utilization_column()
    # test = MachineDataset(single_machine_df, feat, lab)
    # print(test.X)
    
    

    # test_dataset = ForecastDataset(small_df=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    # dataset = GPUDataset(is_training=True, small_df=True)
    # print(dataset.X.shape, dataset.y.shape)
    # print(dataset.X)
