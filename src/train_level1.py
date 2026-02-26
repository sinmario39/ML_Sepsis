import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from preprocessing import load_snapshot, split_train_test, split_level1
from utils import eval_binary, best_threshold_f1

DATA_PATH = "../data/snapshot.csv"
RANDOM_STATE = 42


def build_models():
    # Sostituisce i valori NaN con la mediana della colonna
    imputer = SimpleImputer(strategy="median")

    return {
        "GaussianNB": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        # Pipeline per garantire preprocessing consistente ed evitare leakage
        "DecisionTree(depth=5)": Pipeline([
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


def main():
    df = load_snapshot(DATA_PATH)
    train_df, test_df = split_train_test(df)
    X_train, y_train, X_test, y_test = split_level1(train_df, test_df)

    print("=== LEVEL 1 (Sepsis Vs Non-Sepsis) ===")

    models = build_models()

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Otteniamo probabilità
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Valutazione base (threshold 0.5 implicita)
        eval_binary(y_test, y_pred, y_proba, title=f"{name} | LEVEL 1 (default threshold)")

        # Soglia ottimizzata per F1
        thr, p, r, f1 = best_threshold_f1(y_test, y_proba)
        print(f"\nBest Threshold: {thr:.6f} \nPrecision={p:.4f} \nRecall={r:.4f} \nF1={f1:.4f}")

        custom_pred = (y_proba >= thr).astype(int)
        eval_binary(y_test, custom_pred, title=f"{name} | LEVEL 1 (custom threshold)")

if __name__ == "__main__":
    main()