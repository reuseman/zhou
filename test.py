from pathlib import Path

import numpy as np
import af_zhou as zhou
import utils
import wfdb
import wfdb.processing as processing

import pickle as pkl

# Paths
main_dir_path = Path().absolute()
dataset_dir_path = Path.joinpath(main_dir_path, "dataset")
afdb_dataset = Path().joinpath(dataset_dir_path, "afdb")


save = False
load = True
file_name = Path().joinpath(dataset_dir_path, "afdb_result")


if save:
    array = [
        ["04490", 99, 100, 34.94],
        ["39j48", 99, 100, 34.94],
        ["jf47", 99, 100, 34.94],
    ]
    utils.save_object(file_name, array)

if load:
    with open(file_name, "rb") as f:
        array = pkl.load(f)

    print(array)

file_name_csv = Path().joinpath(dataset_dir_path, "afdb_result.csv")
utils.save_results_csv(file_name_csv, array)