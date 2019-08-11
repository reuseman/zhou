# %%
from pathlib import Path

import utils
import wfdb

# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
afdb_path = Path.joinpath(datasets_path, "afdb")
record_name = "08378"
record_path = str(Path.joinpath(datasets_path, "afdb", record_name))
csv_af_path = Path.joinpath(datasets_path, "af_events.csv")

# Settings
record_len = 9205760  # this sample is not included, because record 0-indexed
entropy_threshold = 0.639
save_array_results = True  # this allows to save the array results on a file

# %%


# Gives back Annotations-QRS and Annotation-AF
def read_record(record_name):
    record_path = str(Path.joinpath(datasets_path, "afdb", record_name))
    annot_qrs = wfdb.rdann(record_path, "qrs")
    annot_af = wfdb.rdann(record_path, "atr")

    return annot_qrs, annot_af


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
        end_value = annot_qrs.sample[i] + 1
        ones = binary_af_samples[start_value:end_value].count(1)

        interval_length = end_value - start_value
        percentage_of_ones = ones / interval_length
        binary_af_qrs.append(percentage_of_ones)

        start_value = annot_qrs.sample[i]

    # [!] TODO move up in the FOR
    return [1 if elem >= entropy_threshold else 0 for elem in binary_af_qrs]


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
correct_af_events = utils.get_af_events(csv_af_path)

afdb_names = set([x.stem for x in afdb_path.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")

results = list()

for record_name in afdb_names:
    record_path = str(Path.joinpath(afdb_path, record_name))
    annot_qrs, annot_af = read_record(record_path)

    patient_af_events = correct_af_events[record_name]

    binary_af_samples = compute_binary_af_samples(patient_af_events)
    binary_af_qrs = compute_binary_af_qrs(binary_af_samples, annot_qrs)

    oracle_af_qrs = compute_oracle_af_qrs(patient_af_events, annot_qrs)

    tp, tn, fp, fn = 0, 0, 0, 0
    for prediction, oracle in zip(binary_af_qrs, oracle_af_qrs):
        if prediction == oracle == 1:
            tp += 1
        elif prediction == oracle == 0:
            tn += 1
        elif prediction != oracle == 0:
            fp += 1
        else:
            fn += 1

    results.append([record_name, tp, tn, fp, fn])

    # se, sp, ppv, acc = compute_metrics(tp, tn, fp, fn)


if save_array_results:
    file_name = Path().joinpath(datasets_path, "afdb_result")
    utils.save_object(file_name, results)
