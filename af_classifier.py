# %%
from pathlib import Path

import utils
import wfdb
import af_zhou as zhou

import time

start_time = time.time()

# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
afdb_path = Path.joinpath(datasets_path, "afdb")
csv_af_path = Path.joinpath(datasets_path, "af_events.csv")

# Settings
record_len = 9205760
entropy_threshold = 0.639
save_array_results = True  # this allows to save the array results on a file

# %%


# Gives back Annotations-QRS and Annotation-AF
def read_record_qrs(record_name):
    record_path = str(Path.joinpath(datasets_path, "afdb", record_name))
    annot_qrs = wfdb.rdann(record_path, "qrs")

    return annot_qrs


# Generates an array of samples matching the record, where 1 is AF and 0 not AF
def compute_binary_af_samples(patient_af_events):
    binary_af_samples = [0] * record_len

    for af_records in patient_af_events:
        binary_af_samples[af_records[0] : af_records[1] + 1] = map(
            lambda x: 1, binary_af_samples[af_records[0] : af_records[1] + 1]
        )

    return binary_af_samples


# Generates an array of qrs, where 1 is an heart beat with AF and 0 not AF
# [!] TODO change implementation to see if improvements are possible
def compute_binary_af_qrs(binary_af_samples, annot_qrs):
    binary_af_qrs = list()

    # With this approach the ends of the range are repeated, except 0 and a[n],
    #   and the percentage based on numbers of 1s is calculated
    start_value = 0
    for i in range(0, len(annot_qrs.sample)):
        # [!] With the commented implementation FN 297 and not 296
        end_value = annot_qrs.sample[i]
        # end_value = annot_qrs.sample[i] + 1

        ones = binary_af_samples[start_value:end_value].count(1)

        interval_length = end_value - start_value
        percentage_of_ones = ones / interval_length

        element = 1 if percentage_of_ones >= entropy_threshold else 0
        binary_af_qrs.append(element)

        start_value = annot_qrs.sample[i] + 1
        # start_value = annot_qrs.sample[i]

    return binary_af_qrs


def compute_oracle_af_qrs(patient_af_events, annot_qrs):
    oracle_af_qrs = list()
    i = 0
    n = len(patient_af_events)
    af_records = patient_af_events[i]
    interval = range(af_records[0], af_records[1] + 1)

    for qrs in annot_qrs.sample:
        if qrs >= interval[-1]:
            if i < n - 1:
                i = i + 1
                af_records = patient_af_events[i]
                interval = range(af_records[0], af_records[1] + 1)
            else:
                oracle_af_qrs.append(0)

        if qrs < interval[0]:
            oracle_af_qrs.append(0)
        elif qrs >= interval[0] and qrs <= interval[-1]:
            oracle_af_qrs.append(1)

    return oracle_af_qrs


# %%
# Read all records, build prediction and oracle an compare them
correct_af_events = utils.get_af_events(csv_af_path)

afdb_names = set([x.stem for x in afdb_path.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")

results = list()
for record_name in afdb_names:
    record_path = str(Path.joinpath(afdb_path, record_name))
    annot_qrs = read_record_qrs(record_path)
    # print("ANNOT QRS LEN: ", len(annot_qrs.sample))

    # Three 0s are added to match the len of QRS points
    entr = zhou.get_entropy(annot_qrs.sample)
    predictions = list(map(lambda x: 1 if x >= entropy_threshold else 0, entr))
    predictions.insert(0, 0)
    predictions.insert(0, 0)
    predictions.insert(0, 0)
    # print("ENTROPY LEN: ", len(predictions))

    patient_af_events = correct_af_events[record_name]

    # binary_af_samples = compute_binary_af_samples(patient_af_events)
    # binary_af_qrs = compute_binary_af_qrs(binary_af_samples, annot_qrs)

    oracle_af_qrs = compute_oracle_af_qrs(patient_af_events, annot_qrs)

    tp, tn, fp, fn = 0, 0, 0, 0
    for prediction, oracle in zip(predictions, oracle_af_qrs):
        if prediction == oracle == 1:
            tp += 1
        elif prediction == oracle == 0:
            tn += 1
        elif prediction != oracle == 0:
            fp += 1
        else:
            fn += 1

    results.append([record_name, tp, tn, fp, fn])
    # se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)

print("EXECUTION TIME: ", time.time() - start_time)

if save_array_results:
    file_name = Path().joinpath(datasets_path, "afdb_result")
    utils.save_object(file_name, results)
