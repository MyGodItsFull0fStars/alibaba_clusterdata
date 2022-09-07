import math
from sklearn.metrics import mean_squared_error


def calculate_root_mean_squared_error(actual_values, predicted_values) -> float:
    """calculate the root mean squared error for calculating the quality of the regression

    Args:
        actual_values (pandas Dataframe): A dataframe with the actual data
        predicted_values (pandas Dataframe): A datagrame with the predicted data


    Returns:
        float: The root mean squared error
    """
    return math.sqrt(mean_squared_error(actual_values, predicted_values))
