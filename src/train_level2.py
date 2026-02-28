from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from preprocessing import load_snapshot, split_train_test, split_level2
from utils import eval_multiclass, tree_importances

DATA_PATH = "/data/snapshot.csv"
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

    X_train, y_train, X_test, y_test = split_level2(train_df, test_df)

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

        eval_multiclass(y_test, y_pred, title=f"{name} | LEVEL 2")

        # Importanze solo per tree
        tree_importances(model, X_train.columns, top_k=15)


if __name__ == "__main__":
    main()