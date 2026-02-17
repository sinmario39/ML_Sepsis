import os
import pandas as pd
from tqdm import tqdm

DATA_PATH = "../data"

print("Working directory:", os.getcwd())
print("DATA_PATH assoluto:", os.path.abspath(DATA_PATH))

def load_all_psv(data_path: str) -> pd.DataFrame:
    psv_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.endswith(".psv"):
                psv_files.append(os.path.join(root, f))

    if not psv_files:
        raise FileNotFoundError(
            f"Nessun file .psv trovato sotto {os.path.abspath(data_path)}. "
            "Controlla che la cartella sia corretta e che i file abbiano estensione .psv."
        )
    """
    print("Trovati file .psv:", len(psv_files))
    print("Esempio primi 5 file:", psv_files[:5])
    """
    all_data = []
    for file_path in tqdm(psv_files, desc="Loading .psv"):
        df = pd.read_csv(file_path, sep="|")
        # patient_id dal nome file (senza estensione)
        df["patient_id"] = os.path.splitext(os.path.basename(file_path))[0]
        # opzionale: da quale split proviene
        df["source_folder"] = os.path.basename(os.path.dirname(file_path))
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def create_snapshot(df):
    # Ordina per paziente e tempo
    df = df.sort_values(["patient_id", "ICULOS"])

    # Prima riga per paziente
    snapshot = df.groupby("patient_id").first().reset_index()

    return snapshot


if __name__ == "__main__":
    print("Loading dataset...")
    full_df = load_all_psv(DATA_PATH)

    print("Creating snapshot...")
    snapshot_df = create_snapshot(full_df)

    print("Snapshot shape:", snapshot_df.shape)
    print(snapshot_df.head())

    print("full_df rows:", len(full_df))
    print("unique patients:", full_df["patient_id"].nunique())
    print("snapshot rows:", len(snapshot_df))

    # Questo ci dà quanti pazienti sono sepsi/non-sepsi nella prima riga.
    print("\nSepsisLabel distribution (snapshot):")
    print(snapshot_df["SepsisLabel"].value_counts(dropna=False))
    print("\nSepsisLabel percentage:")
    print(snapshot_df["SepsisLabel"].value_counts(normalize=True, dropna=False) * 100)

    #Q uesto serve per decidere quali feature tenere (es. teniamo quelle con <80% missing, o una soglia che decideremo).
    missing_pct = snapshot_df.isna().mean().sort_values(ascending=False) * 100
    print("\nMissing % per colonna (snapshot):")
    print(missing_pct)
    # per salvarlo missing_pct.to_csv("../report/missing_snapshot.csv", header=["missing_pct"])

    #Per le colonne più importanti
    core_cols = ["Age", "Gender", "HR", "O2Sat", "SBP", "MAP", "DBP", "Temp", "Resp", "Glucose"]
    core_cols = [c for c in core_cols if c in snapshot_df.columns]

    print("\nDescribe core columns:")
    print(snapshot_df[core_cols].describe(include="all"))

    # Contatori
    snapshot_df["resp_flag"] = (
            (snapshot_df["Resp"] > 22).astype(int) +
            (snapshot_df["O2Sat"] < 94).astype(int) +
            (snapshot_df["HR"] > 100).astype(int)
    )

    snapshot_df["hemo_flag"] = (
            (snapshot_df["SBP"] < 100).astype(int) +
            (snapshot_df["MAP"] < 65).astype(int) +
            (snapshot_df["HR"] > 100).astype(int)
    )

    snapshot_df["metab_flag"] = (
        ((snapshot_df["Glucose"] > 180) | (snapshot_df["Glucose"] < 70)).astype(int)
    )

    print("Resp criteria >=2:", (snapshot_df["resp_flag"] >= 2).sum())
    print("Hemo criteria >=2:", (snapshot_df["hemo_flag"] >= 2).sum())
    print("Metab criteria:", snapshot_df["metab_flag"].sum())

    # Calcolo Overlap
    print("Resp & Hemo overlap:",
          ((snapshot_df["resp_flag"] >= 2) &
           (snapshot_df["hemo_flag"] >= 2)).sum())

    print("Resp & Metab overlap:",
          ((snapshot_df["resp_flag"] >= 2) &
           (snapshot_df["metab_flag"] >= 1)).sum())

    print("Hemo & Metab overlap:",
          ((snapshot_df["hemo_flag"] >= 2) &
           (snapshot_df["metab_flag"] >= 1)).sum())

    # Colonne Macro
    def assign_macro(row):
        if row["SepsisLabel"] == 1:
            return "Sepsis"

        if row["hemo_flag"] >= 2:
            return "Hemodynamic"

        if row["resp_flag"] >= 2:
            return "Respiratory"

        if row["metab_flag"] >= 1:
            return "Metabolic"

        return "Stable"

    # SPLIT
    snapshot_df["macro_label"] = snapshot_df.apply(assign_macro, axis=1)

    print(snapshot_df["macro_label"].value_counts())
    print(snapshot_df["macro_label"].value_counts(normalize=True) * 100)

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        snapshot_df,
        test_size=0.2,
        stratify=snapshot_df["macro_label"],
        random_state=42
    )

    print("Train distribution:")
    print(train_df["macro_label"].value_counts(normalize=True) * 100)

    print("\nTest distribution:")
    print(test_df["macro_label"].value_counts(normalize=True) * 100)

    # PULIZIA DATI
    # Rimozione Colonne Tecniche
    cols_to_drop = ["patient_id", "source_folder", "ICULOS"]
    snapshot_df = snapshot_df.drop(columns=cols_to_drop, errors="ignore")

    # Rimozione Feature con Troppi Missing e Label problematiche
    missing_pct = snapshot_df.isna().mean()

    high_missing_cols = missing_pct[missing_pct > 0.80].index.tolist()
    print("Dropping high missing columns:", high_missing_cols)

    snapshot_df = snapshot_df.drop(columns=high_missing_cols)

    """
    snapshot_df = snapshot_df.drop(
        columns=["resp_flag", "hemo_flag", "metab_flag"],
        errors="ignore"
    )
    """

    # Separazione Target e Feature
    X = snapshot_df.drop(columns=["macro_label", "SepsisLabel"])
    y_macro = snapshot_df["macro_label"]
    y_sepsis = snapshot_df["SepsisLabel"]
    snapshot_df = snapshot_df.drop(
        columns=["resp_flag", "hemo_flag", "metab_flag"],
        errors="ignore"
    )

    # Controllo
    print("Final shape:", snapshot_df.shape)
    print("Remaining columns:", snapshot_df.columns.tolist())













