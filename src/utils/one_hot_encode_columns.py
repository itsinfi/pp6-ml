import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def one_hot_encode_column(df: pd.DataFrame, column: str, mlb: MultiLabelBinarizer):
    encoded = pd.DataFrame(
        mlb.fit_transform(df[column]),
        columns=[f'{column}_{cls}' for cls in mlb.classes_],
        index=df.index
    )
    return encoded

def one_hot_encode_columns(df: pd.DataFrame, columns: list[str]):
    mlb = MultiLabelBinarizer()
    encoded_parts = [one_hot_encode_column(df, col, mlb) for col in columns]
    return pd.concat([df] + encoded_parts, axis=1)