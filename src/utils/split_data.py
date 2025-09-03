import pandas as pd
import hashlib

def split_data(df: pd.DataFrame, train_perc: float = 0.8, val_perc: float = 0.1, test_perc: float = 0.1):
    # check that sum of percentages all equal one together
    if (train_perc + val_perc + test_perc != 1):
        raise 'invalid values for train, value and test percentages (their sum needs to equal 1)'

    # generate a fixed hash
    hashed = df.index.to_series().apply(lambda x: int(hashlib.sha1(str(x).encode()).hexdigest(), 16))

    # shuffle input (will always be the same output)
    df = df.iloc[hashed.argsort()]

    # split dataset
    df_train = df.iloc[:int(train_perc * len(df))]
    df_val = df.iloc[int(train_perc * len(df)):int((train_perc + val_perc) * len(df))]
    df_test = df.iloc[int((train_perc + val_perc) * len(df)):]

    return df_train, df_val, df_test