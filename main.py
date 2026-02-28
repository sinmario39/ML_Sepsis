import argparse
import os
import sys
import runpy

# Importa i moduli da /src
def import_src_to_path():
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Esegue uno script Python dentro src come se fosse lanciato direttamente.
def run_level_script(script_name: str):

    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, "src", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Non trovo {script_path}")
    runpy.run_path(script_path, run_name="__main__")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Sepsis Project - Runner"
    )
    parser.add_argument(
        "--level",
        choices=["1", "2", "all"],
        default="all",
        help="Quale livello eseguire (default: all)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    import_src_to_path()

    if args.level in ("1", "all"):
        print("\n### Running LEVEL 1 (train_level1.py) ###\n")
        run_level_script("train_level1.py")

    if args.level in ("2", "all"):
        print("\n### Running LEVEL 2 (train_level2.py) ###\n")
        run_level_script("train_level2.py")


if __name__ == "__main__":
    main()