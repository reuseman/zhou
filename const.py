from pathlib import Path

# Paths
MAIN_DIR = Path().absolute()

DATASET_DIR = MAIN_DIR / "dataset"
AFDB_DIR = DATASET_DIR / "afdb"
GENERATED_DIR = DATASET_DIR / "generated"
AF_EVENTS_CSV = DATASET_DIR / "af_events.csv"

RESULTS_DIR = MAIN_DIR / "results"
WEKA_RESULTS_DIR = RESULTS_DIR / "weka"
PAPER_RESULTS_DIR = RESULTS_DIR / "paper_replication"
PICKLE_RESULT = PAPER_RESULTS_DIR / "afdb_result"
ZHOU_RESULTS_CSV = PAPER_RESULTS_DIR / "afdb_result_Bc.csv"
PICKLE_GENERATED_DATASET = GENERATED_DIR / "record_data_dict.pkl"


# Settings
RECORD_LEN = 9205760
ENTROPY_THRESHOLD = 0.639
