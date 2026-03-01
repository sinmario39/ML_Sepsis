import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

def eval_binary(y_true, y_pred, y_proba=None, title=None):
    # Stampa confusion matrix + report
    if title:
        print(f"\n=== {title} ===\n")

    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred), "\n")
    print(classification_report(y_true, y_pred, digits=4))

    # Se c'è anche y_proba allora stampa anche ROC-AUC
    if y_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_true, y_proba))

def best_threshold_f1(y_true, y_proba):
    # Trova soglia che massimizza F1 (classe positiva=1). Ritorna threshold, precision, recall ed f1
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # thresholds è lungo len(precision)-1
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)

    i = int(np.argmax(f1))
    return float(thresholds[i]), float(precision[i]), float(recall[i]), float(f1[i])

 # Converte probabilità in predizione 0/1.
def apply_threshold(y_proba, threshold):
    return (np.asarray(y_proba) >= threshold).astype(int)

def eval_multiclass(y_true, y_pred, labels=None, title=None):
    # Stampa macro-F1 + confusion matrix + report
    if title:
        print(f"\n=== {title} ===\n")

    macro = f1_score(y_true, y_pred, average="macro")
    print("Macro-F1:", macro)

    if labels is None:
        labels = sorted(set(y_true))

    print("Confusion matrix (labels order):")
    print(labels)
    print(confusion_matrix(y_true, y_pred, labels=labels), "\n")
    print(classification_report(y_true, y_pred, digits=4))

def tree_importances(pipeline, feature_names, top_k=15):
    # Stampa le feature importanti se il modello è un albero
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_k]

    print("\nTop Feature Importances:")
    for f, imp in pairs:
        print(f"{f:>15s}  {imp:.6f}")

    if hasattr(clf, "get_depth"):
        print("\nTree depth:", clf.get_depth())
    if hasattr(clf, "get_n_leaves"):
        print("Number of leaves:", clf.get_n_leaves())