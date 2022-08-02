import pandas as pd

def get_df(file: str, header=None, sample: bool = False, sample_number: int = 1000):
    if sample:
        df = pd.read_csv(file, header=None).sample(sample_number)
    else:
        df = pd.read_csv(file, header=None)
    
    df.columns = pd.read_csv("{}.header".format(
        file.split('.csv')[0])).columns if header is None else header
    return df