from typing import List, Tuple
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

TRAINING_DATAPATH: str = 'training_df.csv'
TEST_DATAPATH: str = 'test_df.csv'

class GPUDataset(Dataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        batch_size: int = 1000,
        small_df: bool = False
    ) -> None:
        
        data_path = TRAINING_DATAPATH if is_training else TEST_DATAPATH

        # scalers kept as members, since they are also used to invert the transformation
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        self.batch_size = batch_size
        self.small_df = small_df

        self.X, self.y = self.__prepare_dataframe(
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

    def __prepare_dataframe(self, data_path: str = None, data_index: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        data_path = self.__prepare_data_path(data_path)
        data_index = self.__prepare_data_index(data_index)

        df = pd.read_csv(data_path)
        df.set_index(data_index)

        return self.__append_to_feature_and_label_set(df)

    def __prepare_data_path(self, data_path: str) -> str:
        if data_path is None or len(data_path) == 0:
            data_path = 'machine_sorted_df.csv'
        return data_path

    def __prepare_data_index(self, data_index) -> str:
        if data_index is None or len(data_index) == 0:
            data_index = 'start_date'
        return data_index

    def _get_all_machines(self, df: pd.DataFrame) -> np.ndarray:
        return df.machine.unique()

    def __append_to_feature_and_label_set(self, df: pd.DataFrame):
        X_df = pd.DataFrame()
        y_df = pd.DataFrame()

        # df = df.iloc[0:1954000]

        if self.small_df:
            df = df.iloc[0:self.batch_size * 2]

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
        X_df, y_df = self.__scale_dfs(X_df, y_df)

        return self.__transform_dfs_to_tensors(X_df, y_df)

    def __scale_dfs(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_ss = pd.DataFrame(self.standard_scaler.fit_transform(X_df))
        y_mm = pd.DataFrame(self.minmax_scaler.fit_transform(y_df))

        return X_ss, y_mm

    def __transform_dfs_to_tensors(self, X_df, y_df) -> Tuple[torch.Tensor, torch.Tensor]:
        X_df, y_df = X_df.to_numpy(), y_df.to_numpy()

        # Convert to Tensors
        X_df = torch.Tensor(X_df)
        y_df = torch.Tensor(y_df)

        # Reshape Feature Tensor
        X_df = torch.reshape(X_df, (X_df.shape[0], 1, X_df.shape[1]))

        return X_df, y_df

    def __filter_columns(self, X_df, y_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = X_df[self.get_default_feature_columns()]
        y_df = y_df[self.get_default_label_columns()]

        return X_df, y_df


if __name__ == '__main__':
    dataset = GPUDataset(is_training=False, small_df=True)
    print(dataset.X.shape, dataset.y.shape)
