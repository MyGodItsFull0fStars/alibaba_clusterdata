from dataclasses import dataclass, field
from typing import Union
import pandas as pd
import numpy as np


class EvaluationUtils:

    @staticmethod
    def get_evaluation_name_from_path(path: str) -> str:
        return path.split('/')[-1].split('.')[0]

    @staticmethod
    def get_evaluation_key(evaluation_path: str) -> str:
        with_index: int = evaluation_path.index('with')
        return evaluation_path[with_index:-4]

    @staticmethod
    def get_cpu_df(df: pd.DataFrame) -> pd.DataFrame:
        indices: list[int] = [0, 2, 4]
        return df.iloc[:, indices]

    @staticmethod
    def get_mem_df(df: pd.DataFrame) -> pd.DataFrame:
        indices: list[int] = [1, 3, 5]
        return df.iloc[:, indices]

    @staticmethod
    def get_over_allocated_df(df: pd.DataFrame, actual_col, predicted_col) -> pd.DataFrame:
        actual_col = EvaluationUtils.int_to_string_df_column(df, actual_col)
        predicted_col = EvaluationUtils.int_to_string_df_column(
            df, predicted_col)

        over_allocated_indices = df[actual_col] <= df[predicted_col]
        return df[over_allocated_indices]

    @staticmethod
    def get_over_allocated_series(df: pd.DataFrame, actual_col, predicted_col) -> pd.Series:
        oa_df = EvaluationUtils.get_over_allocated_df(
            df, actual_col, predicted_col)
        return oa_df.iloc[:, 1]

    @staticmethod
    def get_over_allocated_df_percentage(df: pd.DataFrame, actual_col, predicted_col) -> float:
        over_allocated_df = EvaluationUtils.get_over_allocated_df(
            df, actual_col, predicted_col)
        return len(over_allocated_df) / len(df)

    @staticmethod
    def get_under_allocated_df(df: pd.DataFrame, actual_col, predicted_col) -> pd.DataFrame:
        actual_col = EvaluationUtils.int_to_string_df_column(df, actual_col)
        predicted_col = EvaluationUtils.int_to_string_df_column(
            df, predicted_col)

        under_allocated_indices = df[actual_col] > df[predicted_col]
        return df[under_allocated_indices]

    @staticmethod
    def get_under_allocated_df_percentage(df: pd.DataFrame, actual_col, predicted_col) -> float:
        under_allocated_df = EvaluationUtils.get_under_allocated_df(
            df, actual_col, predicted_col)
        return len(under_allocated_df) / len(df)

    @staticmethod
    # type: ignore
    def int_to_string_df_column(df: pd.DataFrame, column: Union[int, str]) -> str:
        if type(column) == int:
            return df.columns[column]  # type: ignore
        return column  # type: ignore


@dataclass
class EvaluationScenario:
    name: str
    train_df: pd.DataFrame = field(repr=False)
    test_df: pd.DataFrame = field(repr=False)
    loss_df: pd.DataFrame = field(repr=False)

    def get_cpu_train_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_cpu_df(self.train_df)

    def get_cpu_test_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_cpu_df(self.test_df)

    def get_over_allocated_cpu_train_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_over_allocated_df(self.get_cpu_train_df(), 0, 1)

    def get_over_allocated_cpu_test_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_over_allocated_df(self.get_cpu_test_df(), 0, 1)

    def get_mem_train_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_mem_df(self.train_df)

    def get_mem_test_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_mem_df(self.test_df)

    def get_over_allocated_mem_train_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_over_allocated_df(self.get_mem_train_df(), 0, 1)

    def get_over_allocated_mem_test_df(self) -> pd.DataFrame:
        return EvaluationUtils.get_over_allocated_df(self.get_mem_test_df(), 0, 1)
    
    def get_loss_series(self) -> pd.Series:
        return self.loss_df.iloc[0, :]
    
    def get_training_time_series(self) -> pd.Series:
        return self.loss_df.iloc[1, :]


class EvaluationMetrics:

    __test_df = pd.read_csv('../datasets/training_df.csv', index_col=0)
    actual_cpu: pd.Series = __test_df.iloc[:5000, 1]
    actual_cpu.reset_index(drop=True, inplace=True)

    plan_cpu: pd.Series = __test_df.iloc[:5000, 7]
    plan_cpu.reset_index(drop=True, inplace=True)

    actual_mem: pd.Series = __test_df.iloc[:5000, 4]
    actual_mem.reset_index(drop=True, inplace=True)

    plan_mem: pd.Series = __test_df.iloc[:5000, 8]
    plan_mem.reset_index(drop=True, inplace=True)

    @staticmethod
    def compare_predictions(left_prediction: EvaluationScenario, right_prediction: EvaluationScenario, left_name: str = '', right_name: str = '', is_cpu: bool = True) -> pd.DataFrame:
        left_pred_series: pd.Series
        right_pred_series: pd.Series
        user_alloc_series: pd.Series
        actual_series: pd.Series
        if is_cpu:
            left_pred_series = left_prediction.get_cpu_train_df().iloc[:, 1]
            right_pred_series = right_prediction.get_cpu_train_df().iloc[:, 1]
            user_alloc_series = EvaluationMetrics.plan_cpu
            actual_series = EvaluationMetrics.actual_cpu
        else:  # memory
            left_pred_series = left_prediction.get_mem_train_df().iloc[:, 1]
            right_pred_series = right_prediction.get_mem_train_df().iloc[:, 1]
            user_alloc_series = EvaluationMetrics.plan_mem
            actual_series = EvaluationMetrics.actual_mem

        hardware_key: str = 'CPU' if is_cpu else 'MEM'
        left_key: str = left_name if len(
            left_name) > 0 else left_prediction.name
        right_key: str = right_name if len(
            right_name) > 0 else right_prediction.name

        return pd.concat(
            {
                f'Actual {hardware_key}': actual_series,
                f'{left_key} {hardware_key}': left_pred_series,
                f'{right_key} {hardware_key}': right_pred_series,
                f'User {hardware_key}': user_alloc_series
            }, axis=1)

    @staticmethod
    def get_comparing_metrics_df(df: pd.DataFrame, actual: int = 0, predicted: int = 1, user_alloc: int = 2, table_name: str = 'LSTM Metrics', to_latex: bool = False) -> pd.DataFrame:
        actual_ser: pd.Series = df.iloc[:, actual]
        predicted_ser: pd.Series = df.iloc[:, predicted]
        alloc_ser: pd.Series = df.iloc[:, user_alloc]
        pred_metrics = EvaluationMetrics.get_all_metrics(
            actual_ser, predicted_ser, table_name)
        alloc_metrics = EvaluationMetrics.get_all_metrics(
            actual_ser, alloc_ser, 'User Predicted')

        comp_df = pd.concat([pred_metrics, alloc_metrics])

        if to_latex:
            print(comp_df.round(3).to_latex())
        return comp_df

    @staticmethod
    def get_all_metrics(actual: pd.Series, predicted: pd.Series, table_name: str = 'metrics') -> pd.DataFrame:
        temp_df = pd.concat([actual, predicted], axis=1)
        over_allocation: float = EvaluationUtils.get_over_allocated_df_percentage(
            temp_df, 0, 1) * 100
        temp_dict: dict[str, float] = {
            # 'MSE': EvaluationMetrics.get_mse(actual, predicted),
            'RMSE': EvaluationMetrics.get_rmse(actual, predicted),
            'MAPE': EvaluationMetrics.mean_absolute_percentage_error(actual, predicted),
            'SMAPE': EvaluationMetrics.symmetric_mean_absolute_percentage_error(actual, predicted),
            'OA': over_allocation,
            'UA': 100 - over_allocation
        }

        return pd.DataFrame.from_dict(temp_dict, orient='index', columns=[table_name]).transpose()

    @staticmethod
    def get_mse(actual: pd.Series, predicted: pd.Series) -> float:
        return np.square(np.subtract(actual, predicted)).mean()

    @staticmethod
    def get_rmse(actual: pd.Series, predicted: pd.Series) -> float:
        return np.sqrt(EvaluationMetrics.get_mse(actual, predicted))

    @staticmethod
    def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
        def percentage_error(actual, predicted):
            res = np.empty(actual.shape)
            for j in range(actual.shape[0]):
                if actual[j] != 0:
                    res[j] = (actual[j] - predicted[j]) / actual[j]
                else:
                    res[j] = predicted[j] / np.mean(actual)
            return res
        return np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(predicted)))) * 100

    @staticmethod
    def symmetric_mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
        def percentage_error(actual, predicted) -> np.ndarray:
            result = np.empty(actual.shape)
            for idx in range(actual.shape[0]):
                a, p = actual[idx], predicted[idx]
                if a != 0:
                    result[idx] = (a - p) / ((a + p) / 2)
                else:
                    result[idx] = p / np.mean(actual)
            return result
        return np.mean(np.abs(percentage_error(np.asarray(actual), np.asarray(predicted)))) * 100

    @staticmethod
    def calculate_resource_wastage(actual: pd.Series, predicted: pd.Series) -> float:
        # TODOOOOOOO
        return 0


class EvaluationScenarioCollection:

    evaluation_file_paths: list[str] = list()
    evaluation_collection: dict[str, EvaluationScenario] = dict()

    def __init__(self, evaluation_file_paths: list[str]) -> None:
        self.evaluation_file_paths = evaluation_file_paths
        eval_collection = self._collect_evaluations()
        self._generate_evaluation_scenarios(eval_collection)
        del eval_collection

    def __getitem__(self, item: str) -> EvaluationScenario:
        if item in self.evaluation_collection:
            return self.evaluation_collection[item]
        else:
            return EvaluationScenario('', pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def keys(self):
        return self.evaluation_collection.keys()

    def _collect_evaluations(self) -> dict[str, list[str]]:
        collection_dict: dict[str, list[str]] = dict()

        for evaluation_path in self.evaluation_file_paths:
            evaluation_key = EvaluationUtils.get_evaluation_key(
                evaluation_path)
            if evaluation_key not in collection_dict:
                collection_dict[evaluation_key] = list()
            collection_dict[evaluation_key].append(evaluation_path)

        return collection_dict

    def _generate_evaluation_scenarios(self, evaluation_dict_collection: dict[str, list[str]]) -> None:
        for eval_name, eval_items in evaluation_dict_collection.items():
            train_df, test_df, loss_df = self._get_dataframes_from_collection(
                eval_items)
            self.evaluation_collection[eval_name] = EvaluationScenario(
                eval_name, train_df, test_df, loss_df)

    def _get_dataframes_from_collection(self, col_items: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df: pd.DataFrame = pd.DataFrame()
        test_df: pd.DataFrame = pd.DataFrame()
        loss_df: pd.DataFrame = pd.DataFrame()

        for file_path in col_items:
            if 'train' in file_path:
                train_df = pd.read_csv(file_path, index_col=0)
            elif 'test' in file_path:
                test_df = pd.read_csv(file_path, index_col=0)
            elif 'loss_progression' in file_path:
                loss_df = pd.read_csv(file_path, index_col=0)
        return train_df, test_df, loss_df


if __name__ == '__main__':
    evaluations: list[str] = ['../evaluation/all/util_lstm_train_with_penalty_loss.csv',
                              '../evaluation/all/loss_progression_with_tasks.csv',
                              '../evaluation/all/loss_progression_with_no_tasks.csv',
                              '../evaluation/all/loss_progression_with_small_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_small_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_no_tasks.csv',
                              '../evaluation/all/util_lstm_test_with_small_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_medium_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_instances.csv',
                              '../evaluation/all/loss_progression_with_medium_batch_size.csv',
                              '../evaluation/all/loss_progression_with_penalty_loss.csv',
                              '../evaluation/all/util_lstm_test_with_tasks.csv',
                              '../evaluation/all/util_lstm_train_with_rmse_loss.csv',
                              '../evaluation/all/loss_progression_with_instances.csv',
                              '../evaluation/all/util_lstm_test_with_instances.csv',
                              '../evaluation/all/util_lstm_test_with_no_tasks.csv',
                              '../evaluation/all/loss_progression_with_rmse_loss.csv',
                              '../evaluation/all/util_lstm_test_with_rmse_loss.csv',
                              '../evaluation/all/util_lstm_test_with_large_batch_size.csv',
                              '../evaluation/all/loss_progression_with_large_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_large_batch_size.csv',
                              '../evaluation/all/util_lstm_train_with_tasks.csv',
                              '../evaluation/all/util_lstm_test_with_medium_batch_size.csv',
                              '../evaluation/all/util_lstm_test_with_penalty_loss.csv']

    esc = EvaluationScenarioCollection(evaluations)

    evaluation = esc.evaluation_collection['with_no_tasks']

    print(evaluation)

    print(evaluation.get_cpu_train_df())
