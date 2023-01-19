import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List


def get_over_allocated_series(df: pd.DataFrame, column: int, quantile: float = 0.99) -> pd.Series:
    series = df.iloc[:, column].dropna()
    series = series[series >= 1]
    return get_under_quantile_series(series, quantile)


def get_over_allocated_dataframe(df: pd.DataFrame, columns: List[int], quantile: float = 0.99) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    for col in columns:
        temp_df[df.columns[col]] = get_over_allocated_series(df, col, quantile)
    return temp_df


def describe_over_allocated_dataframe(df: pd.DataFrame, columns: List[int], quantile: float = 0.99) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    for col in columns:
        temp_df[df.columns[col]] = get_over_allocated_series(
            df, col, quantile).describe()
    return temp_df


def get_under_allocated_series(df: pd.DataFrame, column: int, quantile: float = 0.99) -> pd.Series:
    series = df.iloc[:, column].dropna()
    series = series[series < 1]
    return get_over_quantile_series(series, quantile)

def get_over_allocation_percentage(df: pd.DataFrame, column: int) -> float:
    return round(get_over_allocated_series(df, column).shape[0] / df.shape[0] * 100, ndigits=5)

def get_under_allocation_percentage(df: pd.DataFrame, column: int) -> float:
    return round(100 - get_over_allocation_percentage(df, column), ndigits=5)


def get_under_allocated_dataframe(df: pd.DataFrame, columns: List[int], quantile: float = 0.99) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    quantile = 1 - quantile
    for col in columns:
        temp_df[df.columns[col]] = get_under_allocated_series(
            df, col, quantile)
    return temp_df


def describe_under_allocated_dataframe(df: pd.DataFrame, columns: List[int], quantile: float = 0.99) -> pd.DataFrame:
    temp_df = pd.DataFrame()
    quantile = 1 - quantile
    for col in columns:
        temp_df[df.columns[col]] = get_under_allocated_series(
            df, col, quantile).describe()
    return temp_df


def get_over_quantile_series(data: pd.Series, quantile: float = 0.99) -> pd.Series:
    q = data.quantile(quantile)
    data = data[data > q]

    return data


def get_under_quantile_series(data: pd.Series, quantile: float = 0.99) -> pd.Series:
    q = data.quantile(quantile)
    data = data[data < q]

    return data


cycler = plt.cycler(linestyle=['-', '-', '-', '-.', '--', '-', '--'],
                    color=['black', 'blue', 'green',
                           'grey', 'green', 'black', 'green'],
                    )


def plot_df(
    df: pd.DataFrame,
    df_columns: list = None,  # type: ignore
    start_range: int = -1,
    end_range: int = -1,
    save_plot_path: str = None,  # type: ignore
    y_label: str = ''
) -> None:

    if df_columns is None or len(df_columns) == 0:
        df_columns = df.columns.to_list()
    if start_range == -1:
        start_range = 0
    if end_range == -1:
        end_range = len(df)
    fig, ax = plt.subplots()

    plt.xlabel('time step', fontsize=45)
    plt.ylabel(y_label, fontsize=45)

    ax.set_prop_cycle(cycler)
    plot = df[df_columns].iloc[start_range:end_range].plot.line(
        figsize=(25, 20), 
        linewidth=4, 
        fontsize=40, 
        ax=ax)
    
    plt.legend(fontsize=40)
    if save_plot_path is not None:
        plot.figure.savefig(save_plot_path)


def plot_histogram(
    data: pd.Series, 
    bin_size: int = 100, 
    save_path: str = '', 
    over_allocation: bool = True, 
    ) -> None:
    
    if over_allocation:
        cm = plt.cm.get_cmap('PuBuGn_r')
    else:
        cm = plt.cm.get_cmap('PuBuGn')
    # cm = plt.cm.get_cmap('YlGnBu_r')
    # plt.cm.PuBuGn_r

    _, bins, patches = plt.hist(
        data, 
        bin_size, 
        color='green', 
        edgecolor='black'
        )

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    scale_color = bin_centers - min(bin_centers)
    scale_color /= max(scale_color)

    for c, p in zip(scale_color, patches):
        plt.setp(p, 'facecolor', cm(c))

    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (20, 15),
        'axes.labelsize': 25,
        'axes.titlesize': 25,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    }
    plt.rcParams.update(params)

    plt.xlabel('Deviation from Actual Value')
    plt.ylabel('PDF Normalized Occurrence of Value')

    if over_allocation:
        plt.xlim([1, data.max()])
        plt.xticks(range(1, int(data.max() + 1)))

    if len(save_path) > 0:
        plt.savefig(save_path)
        
    # plt.legend(['mean', 'std'], [data.mean(), data.std()])
    plt.show()


def plot_dataframe_kde(
    data: pd.DataFrame,
    fill: bool = False,
) -> None:
    sns.kdeplot(data=data, fill=fill)

def plot_dataframe_histogram(
    data: pd.DataFrame, 
    bin_size: int = 40, 
    save_path: str = '', 
    over_allocation: bool = True, 
    stacked: bool = False,
    density: bool = False,
    ) -> None:
    
    # fig, ax = plt.subplots(figsize=(20, 15))
    
    _, bins, patches = plt.hist(
        data, 
        bin_size, 
        edgecolor='black',
        label=data.columns,
        stacked=stacked,
        density=density,
        )
    
    # sns.kdeplot(data, shade=True)
    
    # data.plot(kind='hist', density=density, bins=bin_size, stacked=stacked, edgecolor='black')
    # data.plot(kind='kde')
    
    plt.xlabel('Deviation from Actual Value (1)')
    plt.ylabel('Frequency of Value')

    if over_allocation:
        min_value = 1
        max_value = int(data.max().max())
        plt.xlim([min_value, max_value])
        plt.xticks(range(min_value, max_value + 1))
    
        
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (20, 15),
        'axes.labelsize': 25,
        'axes.titlesize': 25,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    }
    plt.rcParams.update(params)
        
    plt.legend()
    if len(save_path) > 0:
        plt.savefig(save_path)
    plt.show()

def get_allocation_distribution_df(df: pd.DataFrame, columns: List[int] = None) -> pd.DataFrame:
    if columns is None:
        columns = range(len(df.columns))

        
    percentage_list: list = []
    for col in columns:
        percentage_list.append((get_over_allocation_percentage(df, col), get_under_allocation_percentage(df, col)))
        
    
    return pd.DataFrame(percentage_list, index=df.columns, columns=['over-allocated', 'under-allocated'])


def plot_bars(
    df: pd.DataFrame, 
    figsize: tuple = (20, 12), 
    bar_width: float = 0.3, 
    x_label: str = 'Prediction', 
    y_label: str = 'Percentage'
    ) -> None:
    # source https://medium.com/the-researchers-guide/introduction-to-dodged-bar-plot-matplotlib-pandas-and-seaborn-visualization-guide-part-2-1-49e2fbc9ac39
    # Generating labels and index
    label = df.index
    x = np.arange(len(label))

    #create the base axis
    fig, ax = plt.subplots(figsize=figsize) #set the width of the bars
    width = bar_width # add first pair of bars
    rect1 = ax.bar(x - width / 2,
                df[df.index[0]],
                width = width, 
                label = "No",
                edgecolor = "black") # add second pair of bars 
    rect2 = ax.bar(x + width / 2,
                df[df.index[1]],
                width = width,
                label = "Yes",
                edgecolor = "black")

    # Reset x-ticks
    ax.set_xticks(x)
    # Setting x-axis tick labels
    ax.set_xticklabels(label)

    # Adding bar values
    for p in ax.patches:
        t = ax.annotate(str(p.get_height()) + "%", xy = (p.get_x() + 0.05, p.get_height() + 1))
        t.set(color = "black", size = 20)
        
    # Remove spines
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)# Adding axes and tick labels
    ax.tick_params(axis = "x", labelsize = 18, labelrotation = 0)
    ax.set_ylabel(y_label, size = 25)
    ax.set_xlabel(x_label, size = 25)

    # Customize legend
    ax.legend(labels = df.columns,
            fontsize = 18,
            title = "Allocation",
            title_fontsize = 20)# # Fix legend position
    ax.legend_.set_bbox_to_anchor([1, 1])
