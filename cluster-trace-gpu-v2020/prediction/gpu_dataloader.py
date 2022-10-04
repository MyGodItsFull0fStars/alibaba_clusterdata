from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

TRAINING_DATAPATH: str = 'training_df.csv'
TEST_DATAPATH: str = 'test_df.csv'


class ForecastDataset(Dataset):

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

        # scalers kept as members, since they are also used to invert the transformation
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        self.batch_size: int = batch_size
        self.small_df: bool = small_df

        self.X: Tensor = torch.Tensor()
        self.y: Tensor = torch.Tensor()

        # self.X_orig, self.y_orig = torch.clone(self.X), torch.clone(self.y)
        self.num_samples: int = -1

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
        return ['avg_mem', 'max_mem']

    def _get_gpu_utilization_columns(self) -> List[str]:
        return ['gpu_wrk_util',
                'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']

    def _get_runtime_column(self) -> List[str]:
        return ['runtime']

    def _get_plan_utilization_columns(self) -> List[str]:
        return ['plan_cpu', 'plan_mem', 'plan_gpu']

    def _get_cap_utilization_columns(self) -> List[str]:
        return ['cap_cpu', 'cap_mem', 'cap_gpu']

    def __prepare_data_path(self, data_path: str) -> str:
        if data_path is None or len(data_path) == 0:
            data_path = 'machine_sorted_df.csv'
        return data_path

    def __prepare_data_index(self, data_index) -> str:
        if data_index is None or len(data_index) == 0:
            data_index = 'start_date'
        return data_index

    def _scale_dfs(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_ss = pd.DataFrame(self.standard_scaler.fit_transform(X_df))
        y_mm = pd.DataFrame(self.minmax_scaler.fit_transform(y_df))

        return X_ss, y_mm

    def _resize_df(self, df: pd.DataFrame, split_index: int = 10000) -> pd.DataFrame:
        if self.small_df:
            return df.iloc[:split_index]
        else:
            return df.iloc[:1954000]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if 0 <= index < self.num_samples:
            return self.X[index], self.y[index]


class RuntimeDataset(ForecastDataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        batch_size: int = 1000,
        future_step: int = 10,
        small_df: bool = False
    ) -> None:
        super(RuntimeDataset, self).__init__(is_training, data_index,
                                             batch_size, future_step, small_df)  # type: ignore

        self.X, self.y = self._prepare_data_tensors()

    def _prepare_data_tensors(self) -> Tuple[Tensor, Tensor]:
        df = self._read_csv()
        df = self._resize_df(df)

        return self._init_data_tensors(df=df)

    def _init_data_tensors(self, df: pd.DataFrame) -> Tuple[Tensor, Tensor]:

        X_df = df[self._get_feature_columns()]
        y_df = df[self._get_label_columns()]

        # X_df, y_df = self._scale_dfs(X_df, y_df)
        print(X_df.head(1))
        print(y_df.head(1))

        return self._get_feature_label_tensors()

    def _get_feature_columns(self) -> List[str]:
        return self._get_plan_utilization_columns() + self._get_cap_utilization_columns() + self._get_job_columns()

    def _get_label_columns(self) -> List[str]:
        return self._get_cpu_utilization_columns() + self._get_mem_utilization_columns() + self._get_runtime_column()


class GPUDataset(ForecastDataset):

    def __init__(
        self,
        is_training: bool = True,
        data_index: str = 'start_date',
        batch_size: int = 1000,
        future_step: int = 10,
        small_df: bool = False
    ) -> None:
        super(GPUDataset, self).__init__(is_training,
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
        # X_df = X_df[self.get_default_feature_columns() + self.get_plan_cap_feature_columns()]
        X_df = X_df[self._get_feature_columns()]
        y_df = y_df[self._get_label_columns()]

        return X_df, y_df


if __name__ == '__main__':

    # test_dataset = ForecastDataset()
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    # test_dataset = GPUDataset(small_df=True)
    # print(test_dataset.__class__.__name__,
    #       test_dataset.X.shape, test_dataset.y.shape)
    test_dataset = RuntimeDataset(small_df=True)
    print(test_dataset.__class__.__name__,
          test_dataset.X.shape, test_dataset.y.shape)

    # dataset = GPUDataset(is_training=True, small_df=True)
    # print(dataset.X.shape, dataset.y.shape)
    # print(dataset.X)
