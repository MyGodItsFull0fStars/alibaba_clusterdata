from tkinter.messagebox import NO
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor

MEAN_KEY: str = 'mean'
STD_DEV_KEY: str = 'std'

MIN_KEY: str = 'min'
MAX_KEY: str = 'max'


class DataFrameScaler():

    def __init__(self, df: DataFrame = None, filter_columns=None) -> None:  # type: ignore

        self.filter_columns = filter_columns
        self.df_data_types = None

        if df is not None:
            self.fit(df)

    def fit(self, df: DataFrame) -> None:
        self.df_columns = df.columns
        # preserve datatypes of dataframe
        self.df_data_types = df.dtypes.to_dict()
        self.scaled_df_columns = self.get_scaled_columns(
            df, self.filter_columns)
        self.std_dev_df = self._get_std_mean_df(df)
        self.norm_dev_df = self._get_norm_min_max_df(df)

    def fit_transform_std(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        return self.standardize_df(df)

    def fit_transform_norm(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        return self.normalize_df(df)

    def _get_std_mean_df(self, df: DataFrame) -> DataFrame:
        std_mean_df = DataFrame(
            index=[MEAN_KEY, STD_DEV_KEY], columns=self.scaled_df_columns)

        std_mean_df.at[MEAN_KEY, self.scaled_df_columns] = df[self.scaled_df_columns].apply(
            lambda x: x.mean())
        std_mean_df.at[STD_DEV_KEY, self.scaled_df_columns] = df[self.scaled_df_columns].apply(
            lambda x: x.std())

        return std_mean_df

    def _get_norm_min_max_df(self, df: DataFrame) -> DataFrame:
        norm_min_max_df = DataFrame(
            index=[MIN_KEY, MAX_KEY], columns=self.scaled_df_columns)

        norm_min_max_df.at[MIN_KEY, self.scaled_df_columns] = df[self.scaled_df_columns].apply(
            lambda x: x.min())
        norm_min_max_df.at[MAX_KEY, self.scaled_df_columns] = df[self.scaled_df_columns].apply(
            lambda x: x.max())

        return norm_min_max_df

    def standardize_df(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, std_columns] = df_c.loc[:, std_columns].apply(
            lambda x: ((x - self.std_dev_df.at[MEAN_KEY, x.name]) / self.std_dev_df.at[STD_DEV_KEY, x.name]))
        # in case the standard deviation is 0 (this happens if all values in a column are identical)
        df_c.fillna(0, inplace=True)

        return df_c

    def standardize_df_(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df.loc[:, std_columns] = df.loc[:, std_columns].apply(
            lambda x: ((x - self.std_dev_df.at[MEAN_KEY, x.name]) / self.std_dev_df.at[STD_DEV_KEY, x.name]))

        df.fillna(0, inplace=True)
        
        return df

    def inverse_standardize_df(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, std_columns] = df_c.loc[:, std_columns].apply(
            lambda x: x * self.std_dev_df.at[STD_DEV_KEY, x.name] + self.std_dev_df.at[MEAN_KEY, x.name])

        return df_c.astype(self.df_data_types)  # type: ignore

    def inverse_standardize_df_(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df.loc[:, std_columns] = df.loc[:, std_columns].apply(
            lambda x: x * self.std_dev_df.at[STD_DEV_KEY, x.name] + self.std_dev_df.at[MEAN_KEY, x.name])

        return df

    def inverse_standardize_tensor(self, t: Tensor) -> DataFrame:
        t_df = self.convert_tensor_to_df(t)
        return self.inverse_standardize_df(t_df).astype(self.df_data_types)

    def normalize_df(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, norm_columns] = df_c.loc[:, norm_columns].apply(
            lambda x: (x - self.norm_dev_df.at[MIN_KEY, x.name]) / (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]))

        return df_c

    def normalize_df_(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df.loc[:, norm_columns] = df.loc[:, norm_columns].apply(
            lambda x: (x - self.norm_dev_df.at[MIN_KEY, x.name]) / (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]))

        return df

    def inverse_normalization_df(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df_c = df.copy()

        df_c.loc[:, norm_columns] = df_c.loc[:, norm_columns].apply(
            lambda x: x * (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]) + self.norm_dev_df.at[MIN_KEY, x.name])

        return df_c.astype(self.df_data_types)  # type: ignore

    def inverse_normalization_df_(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns

        df.loc[:, norm_columns] = df.loc[:, norm_columns].apply(
            lambda x: x * (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]) + self.norm_dev_df.at[MIN_KEY, x.name])

        return df

    def get_scaled_columns(self, df: DataFrame, filter_columns) -> List[str]:
        if filter_columns is None:
            return df.columns.to_list()
        return list(filter(lambda column: column not in filter_columns, df.columns))

    def convert_tensor_to_df(self, t: Tensor) -> DataFrame:
        t.squeeze_()  # remove dimensions with size 1
        return DataFrame(data=t.numpy(), columns=self.df_columns)

    def convert_numpy_to_df(self, numpy_arr: np.ndarray) -> DataFrame:
        return DataFrame(data=numpy_arr, columns=self.df_columns)


class DataFrameStdScaler(DataFrameScaler):

    def __init__(self, df: DataFrame = None, filter_columns=None) -> None:  # type: ignore

        self.filter_columns = filter_columns
        self.df_data_types = None

        if df is not None:
            self.fit(df)

    def fit(self, df: DataFrame) -> None:
        self.df_columns = df.columns
        # preserve datatypes of dataframe
        self.df_data_types = df.dtypes.to_dict()
        self.scaled_df_columns = self.get_scaled_columns(
            df, self.filter_columns)
        self.std_dev_df = self._get_std_mean_df(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        self.fit(df)
        return self.standardize_df(df)

    def inverse_transformation(self, data: Tuple[Tensor, DataFrame, np.ndarray]) -> DataFrame:
        temp_df = DataFrame()
        if type(data) == Tensor:
            temp_df = self.convert_tensor_to_df(data)
        elif type(data) == np.ndarray:
            temp_df = self.convert_numpy_to_df(data)
        elif type(data) == DataFrame:
            temp_df = data

        return self.inverse_standardize_df(temp_df).astype(self.df_data_types)
    
    
class DataFrameNormScaler(DataFrameScaler):
    
    def __init__(self, df: DataFrame = None, filter_columns=None) -> None:
        super().__init__(df, filter_columns)
        
    def fit(self, df: DataFrame) -> None:
        self.df_columns = df.columns
        # preserve datatypes of dataframe
        self.df_data_types = df.dtypes.to_dict()
        self.scaled_df_columns = self.get_scaled_columns(
            df, self.filter_columns)
        self.norm_dev_df = self._get_norm_min_max_df(df)

    def inverse_transformation(self, data: Tuple[Tensor, DataFrame, np.ndarray]) -> DataFrame:
        temp_df = DataFrame()
        if type(data) == Tensor:
            temp_df = self.convert_tensor_to_df(data)
        elif type(data) == np.ndarray:
            temp_df = self.convert_numpy_to_df(data)
        elif type(data) == DataFrame:
            temp_df = data

        return self.inverse_normalization_df(temp_df).astype(self.df_data_types)

if __name__ == '__main__':

    arr = np.arange(50)
    arr = arr.reshape(10, 5)
    df = DataFrame(arr)
    # df = df.astype({0: 'float64'})

    # print(df)
    df_scale = DataFrameScaler(df, [0, 4])
    # print(df_scale.std_dev_df)
    # print(df_scale.norm_dev_df)
    df_std = df_scale.standardize_df(df)
    df_norm = df_scale.normalize_df(df)

    asdf = {x: 'int64' for x in range(5)}
    print(df_std)
    # print('---------------')
    # print(df_norm)
    # print('---------------')
    # print(df)
    # print('---------------')
    inv_std_df = df_scale.inverse_standardize_df(df_std)
    print(inv_std_df)

    # print('---------------')
    # print(df_scale._inverse_normalization_df(df_norm))

    # print(df_scale._inverse_standardize_df(df_scale._standardize_df(df)))
