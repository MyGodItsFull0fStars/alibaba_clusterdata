from math import ceil, floor
from posixpath import split
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np


class GPUDataset(Dataset):

    def __init__(
        self,
        data_path: str = None,
        data_index: str = None,
        feature_columns: list = None,
        label_columns: list = None
    ) -> None:

        self.X, self.y = self.__prepare_dataset(
            data_path, data_index, feature_columns, label_columns)
        self.num_samples: int = self.X.shape[0]

    # def __getitem__(self, index) -> T_co:

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.num_samples

    def __prepare_dataset(self, data_path: str, data_index: str, feature_columns: list, label_columns: list) -> tuple:

        feature_columns = self.__prepare_feature_columns(feature_columns)
        label_columns = self.__prepare_label_columns(label_columns)

        df = self.__prepare_dataframe(data_path, data_index)
        # column_set = set(feature_columns).union(label_columns)
        # print(column_set)

        return (torch.Tensor(1), torch.Tensor(1))

    def __prepare_dataframe(self, data_path: str = None, data_index: str = None, drop_columns: List[str] = None) -> pd.DataFrame:
        data_path = self.__prepare_data_path(data_path)
        data_index = self.__prepare_data_index(data_index)

        df = pd.read_csv(data_path)
        df.set_index(data_index)

        # One-Hot Encoding
        dummies = pd.get_dummies(df.task_name)
        df = df.join(dummies)

        # Drop Unused Columns
        drop_columns = self.__prepare_drop_columns(drop_columns)
        df.drop(columns=drop_columns, inplace=True)

        self.__append_to_feature_and_label_set(df)

        self.df = df

        print(df.columns)

        # TODO drop machine column

    def __prepare_drop_columns(self, drop_columns: List[str]) -> List[str]:
        if drop_columns is None or len(drop_columns) == 0:
            drop_columns = ['gpu_type', 'job_name', 'inst_num', 'task_name']
        return drop_columns

    def __prepare_data_path(self, data_path: str) -> str:
        if data_path is None or len(data_path) == 0:
            data_path = 'training_machine_sorted_df.csv'
        return data_path

    def __prepare_data_index(self, data_index) -> str:
        if data_index is None or len(data_index) == 0:
            data_index = 'start_date'
        return data_index

    def __prepare_feature_columns(self, feature_columns: List[str]) -> List[str]:
        if feature_columns is None or len(feature_columns) == 0:
            feature_columns = ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
                               'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime', 'OpenmpiWorker',
                               'OssToVolumeWorker', 'PyTorchWorker', 'tensorflow', 'worker']
        return feature_columns

    def __prepare_label_columns(self, label_columns: List[str]) -> List[str]:
        if label_columns is None or len(label_columns) == 0:
            label_columns = ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
                             'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime']

        return label_columns

    def _get_all_machines(self, df: pd.DataFrame) -> np.ndarray:
        return df.machine.unique()

    def __append_to_feature_and_label_set(self, df: pd.DataFrame, split_ratio: float = 0.5):
        all_machines = self._get_all_machines(df)

        count = 0
        X_df = pd.DataFrame()
        y_df = pd.DataFrame()
        
        for machine in all_machines:
            print(machine)
            machine_query = df.query(f"machine == '{machine}'")

            X_length = floor(len(machine_query) * split_ratio)
            print(f'split {split_ratio}')
            X_df = X_df.append(machine_query.iloc[0:X_length])
            y_df = y_df.append(machine_query[X_length:])
            
            print(machine_query.shape)
            print(X_df.shape, y_df.shape)
            print(X_df.iloc[0], y_df.iloc[0])
            # print()
            
            break

            # print(machine_query)
            count += 1

            if count == 2:
                break


if __name__ == '__main__':
    dataset = GPUDataset()
    print(len(dataset))
