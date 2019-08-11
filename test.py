from pathlib import Path

import numpy as np
import af_zhou as zhou
import utils
import wfdb


def read_record(record_name):
    record_path = str(Path.joinpath(datasets_path, "afdb", record_name))

    record = wfdb.rdrecord(record_path)
    annot_qrs = wfdb.rdann(record_path, "qrs")
    annot_af = wfdb.rdann(record_path, "atr")

    return record, annot_qrs, annot_af


# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
afdb_dataset = Path().joinpath(datasets_path, "afdb")

record, annot_qrs, annot_af = read_record("04908")
entropy, wv = zhou.get_entropy(annot_qrs.sample)
print(len(entropy))
print(len(wv))

