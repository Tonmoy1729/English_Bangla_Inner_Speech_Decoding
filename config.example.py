from pathlib import Path

# Change this to the folder containing your local EEG data.
DATA_ROOT = Path(r"CHANGE_THIS_TO_YOUR_LOCAL_EEG_DATASET_FOLDER")

PROJECT_ROOT = Path(__file__).resolve().parent

ENG_DIR = DATA_ROOT / "English preprocessed"
BN_DIR = DATA_ROOT / "Bangla preprocessed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
