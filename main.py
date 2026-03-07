import argparse
import os
import sys
import runpy

# Importa i moduli da /src
def import_src_to_path():
    project_root = os.path.dirname(os.path.abspath(__file__)) # Prende la posizione del file
    src_path = os.path.join(project_root, "src") # Costruisce il path verso /src
    if src_path not in sys.path:
        sys.path.insert(0, src_path) # Aggiunge /src alla lista di path dove Python prende i moduli

# Esegue uno script Python dentro src come se fosse lanciato direttamente.
def run_level_script(script_name: str):

    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, "src", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Non trovo {script_path}")
    runpy.run_path(script_path, run_name="__main__") # Questo file viene eseguito come se fosse stato lanciato con python


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Sepsis Project - Runner"
    )

    parser.add_argument(
        "--eda",
        action="store_true",
        help="Esegue l'analisi esplorativa dei dati (eda.py)"
    )

    parser.add_argument(
        "--level",
        choices=["1", "2", "all"],
        default="all",
        help="Quale livello eseguire (default: all)"
    )
    return parser.parse_args()

def main():
    args = parse_args() # Legge gli argomenti della linea di comando
    import_src_to_path() # Aggiunge src al path

    # Esecuzione condizionale
    if args.eda:
        print("\n### Running EDA (eda.py) ###\n")
        run_level_script("eda.py")

    if args.level in ("1", "all"):
        print("\n### Running LEVEL 1 (train_level1.py) ###\n")
        run_level_script("train_level1.py")

    if args.level in ("2", "all"):
        print("\n### Running LEVEL 2 (train_level2.py) ###\n")
        run_level_script("train_level2.py")

    print("\n### Execution completed successfully ###\n")

if __name__ == "__main__":
    main()