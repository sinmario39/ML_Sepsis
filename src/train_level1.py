import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import classification_report

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)

DATA_PATH = "../data/snapshot.csv"
RANDOM_STATE = 42


def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) # Legge lo snapshot csv e carica i dati in un DataFrame
    # Scansiona il Data Frame df per vedere se sono presenti le colonne "Sepsis_Label" e "Macro_Label"
    required = {"SepsisLabel", "macro_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Mancano colonne richieste nello snapshot: {missing}")
    return df


def split_data(df: pd.DataFrame):
    # Split stratificato per preservare la distribuzione delle classi (preserva anche l'1% di Sepsi)
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


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test) # Restituisce la classe predetta 0 o 1 (0 = Non Sepsi e 1 = Sepsi)

    # Probabilità per ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1] # Restituisce la probabilità della classe 1 (Sepsi)
        auc = roc_auc_score(y_test, y_proba) # Restituisce le prestazioni del modello
    else:
        y_proba = None
        auc = None

    print(f"\n=== {name} | LEVEL 1: ===\n")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")
    print(classification_report(y_test, y_pred, digits=4)) # Calcola Precision, Recall, F1-Score e Support per poi stampare

    if auc is not None:
        print("ROC-AUC:", auc)

    # Extra utile con class imbalance precision-recall
    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_proba) # Analizza il trade-off tra Precision e Recall trovando la soglia ottimale

def main():
    df = load_snapshot(DATA_PATH)

    train_df, test_df = split_data(df)

    # Level 1 Solo Sepsi
    X_train = train_df.drop(columns=["SepsisLabel", "macro_label"])
    y_train = train_df["SepsisLabel"]

    X_test = test_df.drop(columns=["SepsisLabel", "macro_label"])
    y_test = test_df["SepsisLabel"]

    models = build_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        # Otteniamo probabilità
        y_proba = model.predict_proba(X_test)[:, 1]

        # Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        # Calcolo F1 per ogni soglia
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_idx = np.argmax(f1_scores) # Restituisce l'indice del valore massimo dell'array
        best_threshold = thresholds[best_idx] # Restituisce la soglia ottimale da una lista di candidate
        best_f1 = f1_scores[best_idx] # Assegna il valore F1-Score dall'array prendendolo dall'array f1_scores alla posizione best_idx

        print("=== LEVEL 1 (Sepsis only) ===")
        print("\nBest threshold:", best_threshold)
        print("Best F1:", best_f1)
        print("Precision at best:", precision[best_idx])
        print("Recall at best:", recall[best_idx])

        evaluate_model(name, model, X_test, y_test)

        # Usiamo la nuova soglia
        custom_pred = (y_proba >= best_threshold).astype(int)

        print("\n=== Custom threshold results ===")
        print(classification_report(y_test, custom_pred, digits=4))


if __name__ == "__main__":
    main()
