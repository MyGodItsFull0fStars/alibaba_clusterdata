from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch import Tensor


MEAN_KEY: str = 'mean'
STD_DEV_KEY: str = 'std'

MIN_KEY: str = 'min'
MAX_KEY: str = 'max'


class DataFrameScaler():

    def __init__(self, df: DataFrame = None, filter_columns=None) -> None:  # type: ignore

        self.filter_columns = filter_columns

        if df is not None:
            self.df_columns = df.columns
            self.df_data_types = df.dtypes.to_dict()
            self.scaled_df_columns = self._get_scaled_columns(
                df, filter_columns)

            self.fit(df)

    def fit(self, df: DataFrame) -> None:
        self.std_dev_df = self._get_std_mean_df(df)
        self.norm_dev_df = self._get_norm_min_max_df(df)

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

    def _standardize_df(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, std_columns] = df_c.loc[:, std_columns].apply(
            lambda x: ((x - self.std_dev_df.at[MEAN_KEY, x.name]) / self.std_dev_df.at[STD_DEV_KEY, x.name]))

        
        return df_c

    def _standardize_df_(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df.loc[:, std_columns] = df.loc[:, std_columns].apply(
            lambda x: ((x - self.std_dev_df.at[MEAN_KEY, x.name]) / self.std_dev_df.at[STD_DEV_KEY, x.name]))

        return df

    def _inverse_standardize_df(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, std_columns] = df_c.loc[:, std_columns].apply(
            lambda x: x * self.std_dev_df.at[STD_DEV_KEY, x.name] + self.std_dev_df.at[MEAN_KEY, x.name])

        return df_c.astype(self.df_data_types)

    def _inverse_standardize_df_(self, df: DataFrame) -> DataFrame:
        std_columns = self.scaled_df_columns
        df.loc[:, std_columns] = df.loc[:, std_columns].apply(
            lambda x: x * self.std_dev_df.at[STD_DEV_KEY, x.name] + self.std_dev_df.at[MEAN_KEY, x.name])

        return df

    def _inverse_standardize_tensor(self, t: Tensor) -> DataFrame:
        t_df = self._convert_tensor_to_df(t)
        return self._inverse_standardize_df(t_df).astype(self.df_data_types)

    def _normalize_df(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df_c = df.copy()
        df_c.loc[:, norm_columns] = df_c.loc[:, norm_columns].apply(
            lambda x: (x - self.norm_dev_df.at[MIN_KEY, x.name]) / (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]))

        return df_c

    def _normalize_df_(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df.loc[:, norm_columns] = df.loc[:, norm_columns].apply(
            lambda x: (x - self.norm_dev_df.at[MIN_KEY, x.name]) / (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]))

        return df

    def _inverse_normalization_df(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns
        df_c = df.copy()

        df_c.loc[:, norm_columns] = df_c.loc[:, norm_columns].apply(
            lambda x: x * (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]) + self.norm_dev_df.at[MIN_KEY, x.name])

        return df_c.astype(self.df_data_types)  # type: ignore

    def _inverse_normalization_df_(self, df: DataFrame) -> DataFrame:
        norm_columns = self.scaled_df_columns

        df.loc[:, norm_columns] = df.loc[:, norm_columns].apply(
            lambda x: x * (self.norm_dev_df.at[MAX_KEY, x.name] - self.norm_dev_df.at[MIN_KEY, x.name]) + self.norm_dev_df.at[MIN_KEY, x.name])

        return df

    def _convert_tensor_to_df(self, t: Tensor) -> DataFrame:
        t.squeeze_()  # remove dimensions with size 1
        return DataFrame(data=t.numpy(), columns=self.df_columns)

    def _get_scaled_columns(self, df: DataFrame, filter_columns) -> List[str]:
        return list(filter(lambda column: column not in filter_columns, df.columns))


if __name__ == '__main__':

    arr = np.arange(50)
    arr = arr.reshape(10, 5)
    df = DataFrame(arr)
    df = df.astype({0: 'float64'})


    # print(df)
    df_scale = DataFrameScaler(df, [0, 4])
    # print(df_scale.std_dev_df)
    # print(df_scale.norm_dev_df)
    df_std = df_scale._standardize_df(df)
    df_norm = df_scale._normalize_df(df)

    
    asdf = {x: 'int64' for x in range(5)}
    print(df_std)
    # print('---------------')
    # print(df_norm)
    # print('---------------')
    # print(df)
    # print('---------------')
    inv_std_df = df_scale._inverse_standardize_df(df_std)
    print(inv_std_df)
       
    # print('---------------')
    # print(df_scale._inverse_normalization_df(df_norm))

    # print(df_scale._inverse_standardize_df(df_scale._standardize_df(df)))