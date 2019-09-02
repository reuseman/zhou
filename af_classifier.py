# %%
from pathlib import Path

import utils
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
read_corrected = True
save_array_results = True  # this allows to save the array results on a file


# %%
# Read all records, build prediction and oracle an compare them
correct_af_events = utils.get_af_events(csv_af_path)

afdb_names = set([x.stem for x in afdb_path.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")

results = list()
af_beats = 0
non_af_beats = 0
hybrid = 0

for record_name in afdb_names:
    annot_qrs = utils.read_record_qrs(afdb_path, record_name, read_corrected)
    # print("ANNOT QRS LEN: ", len(annot_qrs.sample))

    entr = zhou.get_entropy(annot_qrs.sample)
    predictions = list(map(lambda x: 1 if x >= entropy_threshold else 0, entr))
    for e in predictions:
        if e == 0:
            non_af_beats += 1
        else:
            af_beats += 1
    # print("ENTROPY LEN: ", len(predictions))

    patient_af_events = correct_af_events[record_name]

    binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
    oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(0, len(predictions)):
        if oracle[i + 2] != 1 and oracle[i + 2] != 0:
            hybrid += 1
        else:
            if predictions[i] == oracle[i + 2] == 1:
                tp += 1
            elif predictions[i] == oracle[i + 2] == 0:
                tn += 1
            elif predictions[i] != oracle[i + 2] == 0:
                fp += 1
            else:
                fn += 1

    results.append([record_name, tp, tn, fp, fn, len(oracle)])

print("EXECUTION TIME: ", time.time() - start_time)
print("AF BEATS: ", af_beats)
print("NON AF BEATS: ", non_af_beats)
print("HYBID NOT CLASSIFIED: ", hybrid)

if save_array_results:
    file_name = Path().joinpath(datasets_path, "afdb_result")
    utils.save_object(file_name, results)
