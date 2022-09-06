from math import ceil, floor
from tkinter import Variable
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class GPUDataset(Dataset):

    def __init__(
        self,
        data_path: str = None,
        data_index: str = None,
        feature_columns: list = None,
        label_columns: list = None
    ) -> None:

        self.X, self.y = self.__prepare_dataset(
            data_path, data_index)
        self.num_samples: int = self.X.shape[0]

    def __getitem__(self, index):
        if 0 <= index < self.num_samples:
            return self.X[index], self.y[index]

    def __len__(self):
        return self.num_samples

    def get_default_feature_columns(self) -> List[str]:
        return ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
                'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime', 'BladeMain',
                'JupyterTask', 'OpenmpiWorker', 'OssToVolumeWorker', 'PyTorchWorker',
                'TVMTuneMain', 'chief', 'evaluator', 'ps', 'tensorflow', 'worker',
                'xComputeWorker']

    def get_default_label_columns(self) -> List[str]:
        return ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
                             'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'runtime']

    def __prepare_dataset(self, data_path: str, data_index: str) -> tuple:

        return self.__prepare_dataframe(data_path, data_index)

    def __prepare_dataframe(self, data_path: str = None, data_index: str = None, drop_columns: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        df.sort_index(inplace=True)

        return self.__append_to_feature_and_label_set(df)

    def __prepare_drop_columns(self, drop_columns: List[str]) -> List[str]:
        if drop_columns is None or len(drop_columns) == 0:
            drop_columns = ['gpu_type', 'job_name',
                            'inst_num', 'task_name', 'machine']
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
            feature_columns = self.get_default_feature_columns()
        return feature_columns

    def __prepare_label_columns(self, label_columns: List[str]) -> List[str]:
        if label_columns is None or len(label_columns) == 0:
            label_columns = self.get_default_label_columns()

        return label_columns

    def _get_all_machines(self, df: pd.DataFrame) -> np.ndarray:
        return df.machine.unique()

    def __append_to_feature_and_label_set(self, df: pd.DataFrame, batch_size: int = 500):
        X_df = pd.DataFrame()
        y_df = pd.DataFrame()

        # df = df.iloc[0:1954000]
        df = df.iloc[0:5000]

        for step in range(0, len(df) // batch_size, 2):

            feature_start_index = step * batch_size
            feature_end_index = feature_start_index + batch_size

            label_start_index = feature_end_index
            label_end_index = feature_end_index + batch_size

            if label_end_index > len(df):
                break

            X_df = pd.concat(
                [X_df, df.iloc[feature_start_index:feature_end_index]])
            y_df = pd.concat(
                [y_df, df.iloc[label_start_index:label_end_index]])

        X_df = X_df[self.get_default_feature_columns()]
        y_df = y_df[self.get_default_label_columns()]

        X_df, y_df = self.__scale_dfs(X_df, y_df)
        X_df, y_df = X_df.to_numpy(), y_df.to_numpy()

        # Convert to Tensors
        X_df = torch.Tensor(X_df)
        y_df = torch.Tensor(y_df)

        # Reshape Feature Tensor
        X_df = torch.reshape(X_df, (X_df.shape[0], 1, X_df.shape[1]))

        return X_df, y_df

    def __scale_dfs(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        X_ss = pd.DataFrame(standard_scaler.fit_transform(X_df))
        y_mm = pd.DataFrame(minmax_scaler.fit_transform(y_df))

        return X_ss, y_mm


if __name__ == '__main__':
    dataset = GPUDataset()
    
    print(dataset.X.shape, dataset.y.shape)
