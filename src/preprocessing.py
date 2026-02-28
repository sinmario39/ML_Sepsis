import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Caricamento snapshot
# Reinterpreta il path rispetto alla root del progetto
def load_snapshot(path: str) -> pd.DataFrame:
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path.lstrip("/\\"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot non trovato: {path}")

    df = pd.read_csv(path)
    required_cols = {"SepsisLabel", "macro_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne richieste nello snapshot: {missing}")
    return df


# Split Train/Test stratificato
def split_train_test(df: pd.DataFrame, test_size: float = 0.2):

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["macro_label"],
        random_state=RANDOM_STATE
    )

    return train_df, test_df


# Split per LEVEL 1 (Sepsi vs Non-Sepsi)
def split_level1(train_df: pd.DataFrame, test_df: pd.DataFrame):

    X_train = train_df.drop(columns=["SepsisLabel", "macro_label"])
    y_train = train_df["SepsisLabel"]

    X_test = test_df.drop(columns=["SepsisLabel", "macro_label"])
    y_test = test_df["SepsisLabel"]

    return X_train, y_train, X_test, y_test


# Split per LEVEL 2 (Classificazione Macro Labels)
def split_level2(train_df: pd.DataFrame, test_df: pd.DataFrame):

    train_non_sepsis = train_df[train_df["SepsisLabel"] == 0]
    test_non_sepsis = test_df[test_df["SepsisLabel"] == 0]

    X_train = train_non_sepsis.drop(columns=["SepsisLabel", "macro_label"])
    y_train = train_non_sepsis["macro_label"]

    X_test = test_non_sepsis.drop(columns=["SepsisLabel", "macro_label"])
    y_test = test_non_sepsis["macro_label"]

    return X_train, y_train, X_test, y_test