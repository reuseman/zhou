from const import AFDB_DIR, AFDB_EVENTS_CSV, LOCAL_PREDICTION_RESULTS_AFDB
from pathlib import Path

import time
import utils
import zhou
import csv
import numpy as np


# [Settings]
# allows to read the .qrsc when available
read_corrected = True
# allows to save the array results on a file
#  that will be uses by zhou_results.py

start_time = time.time()

# Read all records, build prediction and oracle, than compare them
correct_af_events = utils.get_af_events(AFDB_EVENTS_CSV)

afdb_names = set([x.stem for x in AFDB_DIR.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")

hybrid = 0

# Generates all values in the interval [0.000, 1.000]
all_threshold = np.linspace(0, 1, 1001)

for record_name in sorted(list(afdb_names)):
    record_time = time.time()
    print("Current record: ", record_name)
    results = list()

    annot_qrs = utils.read_record_qrs(AFDB_DIR, record_name, read_corrected)

    patient_af_events = correct_af_events[record_name]
    binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
    oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)

    entr = zhou.get_entropy(annot_qrs.sample)

    entr = entr[126:]
    oracle = oracle[128:]

    for current_threshold in all_threshold:
        predictions = list(map(lambda x: 1 if x >= current_threshold else 0, entr))

        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(0, len(predictions)):
            if oracle[i] != 1 and oracle[i] != 0:
                hybrid += 1
            else:
                if predictions[i] == oracle[i] == 1:
                    tp += 1
                elif predictions[i] == oracle[i] == 0:
                    tn += 1
                elif predictions[i] != oracle[i] == 0:
                    fp += 1
                else:
                    fn += 1

        se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)
        results.append([current_threshold, tp, tn, fp, fn, se, sp, ppv, acc])

    print("     hybrid ", hybrid)
    print("     writing results...")
    file_name = record_name + ".csv"
    csv_path = LOCAL_PREDICTION_RESULTS_AFDB / file_name

    with open(csv_path, mode="w") as results_file:
        fieldnames = ["THRESHOLD", "TP", "TN", "FP", "FN", "SE", "SP", "PPV", "ACC"]
        
        results_writer = csv.writer(
            results_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )

        results_writer.writerow(fieldnames)
        results_writer.writerows(results)

    print("     finished after ", time.time() - record_time)

print("EXECUTION TIME: ", time.time() - start_time)