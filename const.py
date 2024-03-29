from pathlib import Path

# Paths
MAIN_DIR = Path().absolute()

DATASET_DIR = MAIN_DIR / "dataset"
AFDB_DIR = DATASET_DIR / "afdb"
LTAFDB_DIR = DATASET_DIR / "ltafdb"
GENERATED_DIR = DATASET_DIR / "generated"
AFDB_EVENTS_CSV = DATASET_DIR / "afdb_events.csv"
LTAFDB_EVENTS_CSV = DATASET_DIR / "ltafdb_events.csv"

RESULTS_DIR = MAIN_DIR / "results"
WEKA_RESULTS_DIR = RESULTS_DIR / "weka"
LOCAL_PREDICTION_RESULTS_DIR = RESULTS_DIR / "local_prediction"
LOCAL_PREDICTION_RESULTS_AFDB = LOCAL_PREDICTION_RESULTS_DIR / "afdb"
LOCAL_PREDICTION_RESULTS_LTAFDB = LOCAL_PREDICTION_RESULTS_DIR / "ltafdb"
PAPER_RESULTS_DIR = RESULTS_DIR / "paper_replication"
PICKLE_RESULT = PAPER_RESULTS_DIR / "afdb_result"
ZHOU_RESULTS_CSV = PAPER_RESULTS_DIR / "afdb_result_Bc.csv"
PICKLE_GENERATED_DATASET = GENERATED_DIR / "record_data_dict.pkl"


# Settings
RECORD_LEN = 9205760
ENTROPY_THRESHOLD = 0.639
