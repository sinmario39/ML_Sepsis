import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

DATA_PATH = "../data/snapshot.csv"
RANDOM_STATE = 42

def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) # Legge lo snapshot csv e carica i dati in un Data Frame
    # Scansiona il Data Frame df per vedere se sono presenti le colonne "Sepsis_Label" e "Macro_Label"
    required = {"SepsisLabel", "macro_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Mancano colonne richieste nello snapshot: {missing}")
    return df

def split_data(df: pd.DataFrame):
    # Split stratificato per preservare la distribuzione delle classi
    return train_test_split(df, test_size=0.2, stratify=df["macro_label"],random_state=RANDOM_STATE)

def build_models():
    # Sostituisce i valori NaN con la mediana della colonna
    imputer = SimpleImputer(strategy="median")

    models = {
        "GaussianNB": Pipeline(
            [("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())]),
        # Pipeline per garantire preprocessing consistente ed evitare leakage
        "DecisionTree": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_depth=5,
                min_samples_split=50
            ))
        ])
    }
    return models

def main():
    df = load_snapshot(DATA_PATH)

    train_df, test_df = split_data(df)

    # Livello 2: Solo Non-Sepsi
    train_df = train_df[train_df["SepsisLabel"] == 0].copy()
    test_df = test_df[test_df["SepsisLabel"] == 0].copy()
    # Vogliamo solo le macro_label
    # Rimuoviamo eventuali sepsi rimaste per sicurezza
    train_df = train_df[train_df["macro_label"] != "Sepsis"]
    test_df = test_df[test_df["macro_label"] != "Sepsis"]

    # Features
    X_train = train_df.drop(columns=["SepsisLabel", "macro_label"])
    y_train = train_df["macro_label"]

    X_test = test_df.drop(columns=["SepsisLabel", "macro_label"])
    y_test = test_df["macro_label"]

    print("=== LEVEL 2 (Non-Sepsis only) ===")
    print("Train size:", X_train.shape, " Test size:", X_test.shape)
    print("\nTrain label distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))
    print("\nTest label distribution (%):")
    print((y_test.value_counts(normalize=True) * 100).round(2))

    models = build_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n=== {name} | LEVEL 2 ===\n")
        print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
        print("Confusion matrix:")
        labels = sorted(y_test.unique())
        print(labels)
        print(confusion_matrix(y_test, y_pred, labels=labels), "\n")
        print(classification_report(y_test, y_pred, digits=4))

        # Feature importance e info albero solo per il DecisionTree
        clf = model.named_steps["clf"] # Estrai il classificatore dalla pipeline
        if hasattr(clf, "feature_importances_"):
            feature_names = X_train.columns
            importances = clf.feature_importances_

            # Andiamo a costruire un Data Frame ordinato
            feat_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            print("\nTop 15 Feature Importances:")
            print(feat_imp.head(15))

            print("\nTree depth:", clf.get_depth())
            print("Number of leaves:", clf.get_n_leaves())

if __name__ == "__main__":
    main()
