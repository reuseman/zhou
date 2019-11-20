from const import AFDB_DIR, PICKLE_RESULT, AFDB_EVENTS_CSV, ENTROPY_THRESHOLD

import pickle as pkl
import time

import utils
import zhou


# [Settings]
# allows to read the .qrsc when available
read_corrected = True
# allows to save the array results on a file
#  that will be uses by zhou_results.py
save_array_results = True

start_time = time.time()

# Read all records, build prediction and oracle, than compare them
correct_af_events = utils.get_af_events(AFDB_EVENTS_CSV)

afdb_names = set([x.stem for x in AFDB_DIR.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")

results = list()
af_beats = 0
non_af_beats = 0
hybrid = 0

for record_name in ["07162"]:
    annot_qrs = utils.read_record_qrs(AFDB_DIR, record_name, read_corrected)
    # print("ANNOT QRS LEN: ", len(annot_qrs.sample))

    entr = zhou.get_entropy(annot_qrs.sample)
    predictions = list(map(lambda x: 1 if x >= ENTROPY_THRESHOLD else 0, entr))
    for e in predictions:
        if e == 0:
            non_af_beats += 1
        else:
            af_beats += 1
    # print("ENTROPY LEN: ", len(predictions))

    patient_af_events = correct_af_events[record_name]
    binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
    oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)
    # print("ORACLE LEN: ", len(oracle))

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
    with open(PICKLE_RESULT, "wb") as f:
        pkl.dump(results, f)
