from pathlib import Path
from scipy.signal import detrend
from spectrum import aryule
from const import (
    AFDB_DIR,
    GENERATED_DIR,
    AFDB_EVENTS_CSV,
    ENTROPY_THRESHOLD,
    PICKLE_GENERATED_DATASET,
)

import numpy as np
import pandas as pd
import pickle as pkl
import wfdb

import zhou
import utils
import time

# Settings
read_corrected = True
save_records_data = True

add_fft = False
add_ar = False
fft_n = 16
ar_n = 4
blocks = 2

# Paths
GENERATED_DIR.mkdir(exist_ok=True)
folder_to_create = GENERATED_DIR / "encoded_entropy_126_removed"
folder_to_create.mkdir(exist_ok=True)


def read_heart_signal(record_name):
    record_dir = str(Path.joinpath(AFDB_DIR, record_name))
    return wfdb.rdrecord(record_dir, channel_names=["ECG1"]).p_signal.squeeze()


def compute_unimol_entropy(bpm_list):
    # Stable BPMs are labeled as 1
    for i in range(0, len(bpm_list)):
        if bpm_list[i] <= 50:
            bpm_list[i] = 0
        elif bpm_list[i] >= 120:
            bpm_list[i] = 2
        else:
            bpm_list[i] = 1

    unimol_entropy = list()

    # Count the number of 1s in a window of 10 elements
    for i in range(9, len(bpm_list)):
        start_index = i - 9
        end_index = i
        entropy = bpm_list[start_index : end_index + 1].count(1)
        unimol_entropy.append(entropy)

    return unimol_entropy


def divide_signal(signal, number_of_blocks=2):
    if number_of_blocks <= 0:
        return []

    block_dimension = int(len(signal) / number_of_blocks)
    reminder = len(signal) % number_of_blocks
    if block_dimension == 0:
        return [signal]

    blocks = list()
    for i in range(0, len(signal), block_dimension):
        blocks.append(signal[i : i + block_dimension])

    if reminder:
        blocks[-2] = np.concatenate((blocks[-2], blocks[-1]))
        blocks.pop()

    return blocks


def get_arff_head(relation_name, add_fft=False, add_ar=False):
    arff_head = "@relation {}\n".format(relation_name)
    arff_head = "".join(
        [
            arff_head,
            "@attribute zhou_encoded_entr numeric\n@attribute unim_entr numeric\n",
        ]
    )

    if add_fft:
        for i in range(1, blocks + 1):
            for j in range(1, fft_n + 1):
                arff_head = "".join(
                    [arff_head, "@attribute fft{}_b{} numeric\n".format(j, i)]
                )

    if add_ar:
        for i in range(1, blocks + 1):
            for j in range(1, ar_n + 1):
                arff_head = "".join(
                    [arff_head, "@attribute ar{}_b{} numeric\n".format(j, i)]
                )

    return "".join([arff_head, "@attribute physionet {0.0,1.0}\n@data\n"])


def get_arff_data(record):
    df = pd.DataFrame(record)
    data = df.to_csv(index=False, header=False)

    return data


start_time = time.time()
print("Start time: ", start_time)

correct_af_events = utils.get_af_events(AFDB_EVENTS_CSV)

afdb_names = set([x.stem for x in AFDB_DIR.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")
if add_fft or add_ar:
    afdb_names.remove("03665")
    afdb_names.remove("00735")

records_data = dict()

# Se esiste file pickle non eseguire il For sottostante
execute_computation = True
if Path.exists(PICKLE_GENERATED_DATASET):
    execute_computation = False
    with open(PICKLE_GENERATED_DATASET, "rb") as f:
        records_data = pkl.load(f)
    print("[!] Record_data: dict already exists, computation will be skipped.")

if execute_computation:
    print("[!] Computation will be executed.")
    for record_name in afdb_names:
        annot_qrs = utils.read_record_qrs(AFDB_DIR, record_name, read_corrected)
        print("LEN QRS: ", len(annot_qrs.sample))

        # COMPUTE ENTROPY
        bpm_list = zhou.compute_bpm(annot_qrs.sample)
        print("LEN BPM: ", len(bpm_list))
        sy_list = zhou.compute_sy(bpm_list)
        wv_list = zhou.compute_wv(sy_list)
        entr = zhou.compute_entropy(wv_list)
        predictions = list(map(lambda x: 1 if x >= ENTROPY_THRESHOLD else 0, entr))
        print("LEN ENTROPY: ", len(entr))
        print("LEN PREDICTIONS: ", len(predictions))

        # COMPUTE UNIMOL ONE
        unimol_entropy = compute_unimol_entropy(bpm_list)
        print("LEN UNIMOL ENTROPY: ", len(unimol_entropy))

        # COMPUTE FOURIER AND/OR AR COEFFICIENTS
        #   numpy arrays of shape (fft_n,) and/or (ar_n,) are added
        #    to the fft_values and/or ar_values
        if add_fft or add_ar:
            fft_values = list()
            ar_values = list()

            signal = read_heart_signal(record_name)
            constant_detrend = detrend(signal, type="constant")

            for i in range(0, len(annot_qrs.sample) - 1):
                rr_signal = constant_detrend[
                    annot_qrs.sample[i] : annot_qrs.sample[i + 1]
                ]
                # divided_s = divide_signal(rr_signal, number_of_blocks=blocks)
                divided_signal = np.array_split(rr_signal, blocks)

                fft_blocks = list()
                ar_blocks = list()
                for j in range(0, len(divided_signal)):
                    if add_fft:
                        fft_blocks.append(np.fft.fft(divided_signal[j], fft_n).real)

                    if add_ar:
                        ar_blocks.append(aryule(divided_signal[j], ar_n)[0])

                if add_fft:
                    # fft_blocks = [array([1,2,3]), array([4,5,6])] ->
                    #   array([1,2,3,4,5,6])
                    fft_values.append(np.asarray(fft_blocks).flatten())

                if add_ar:
                    ar_values.append(np.asarray(ar_blocks).flatten())

            print("LEN FFT: ", len(fft_values))
            print("LEN AR: ", len(ar_values))

        # COMPUTE ORACLE
        patient_af_events = correct_af_events[record_name]
        binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
        oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)
        print("LEN ORACLE: ", len(oracle))

        entr = np.asarray(entr[126:])
        unimol_entropy = np.asarray(unimol_entropy[119:])
        if add_fft:
            # The shape in the beginning is (n, fft_n)
            fft_values = np.asarray(fft_values[9:]).transpose()
        if add_ar:
            # The shape in the beginning is (n, ar_n)
            ar_values = np.asarray(ar_values[9:]).transpose()
        oracle = np.asarray(oracle[128:])

        """ entr = np.asarray(entr[7:])
        unimol_entropy = np.asarray(unimol_entropy)
        if add_fft:
            # The shape in the beginning is (n, fft_n)
            fft_values = np.asarray(fft_values[9:]).transpose()
        if add_ar:
            # The shape in the beginning is (n, ar_n)
            ar_values = np.asarray(ar_values[9:]).transpose()
        oracle = np.asarray(oracle[9:]) """

        print("\nLENGTH IN THE RECORDS DATA: ")
        print("ENTR: ", entr.shape)
        print("UNIM: ", unimol_entropy.shape)
        if add_fft:
            print("FFT_: ", fft_values.shape)
        if add_ar:
            print("AR_V: ", ar_values.shape)
        print("ORAC: ", oracle.shape)

        records_data[record_name] = np.asarray([entr, unimol_entropy])
        temp = np.asarray([entr, unimol_entropy])
        if add_fft:
            # Axis = 0, means direction from top to bottom (collapses rows)
            temp = np.append(temp, [*fft_values], axis=0)
        if add_ar:
            temp = np.append(temp, [*ar_values], axis=0)
        temp = np.append(temp, [oracle], axis=0)
        records_data[record_name] = temp.transpose()

    if save_records_data:
        with open(PICKLE_GENERATED_DATASET, "wb") as f:
            pkl.dump(records_data, f)
        print("[!] Record_data:dict created.")

print("\n")

# GENERATE FOLDERS WITH DATASET
for record_name in afdb_names:
    print("Current record: ", record_name)
    # Folder's name is the name of the record wich is used to test
    sub_folder_record = Path.joinpath(folder_to_create, record_name)
    sub_folder_record.mkdir(exist_ok=True)
    
    # Create Test.arff using record_name
    test_path = Path.joinpath(sub_folder_record, "test.arff")

    arff_head = get_arff_head("test", add_fft, add_ar)
    arff_data = get_arff_data(records_data[record_name])
    arff = "".join([arff_head, arff_data])

    with open(test_path, "w") as f:
        f.write(arff)
    print(" Test set created")

    # Create Train.arff using every record except record_name
    train_path = Path.joinpath(sub_folder_record, "train.arff")

    arff_head = get_arff_head("train", add_fft, add_ar)
    arff_data = ""
    for record in afdb_names:
        if record == record_name:
            continue
        arff_data = "".join([arff_data, get_arff_data(records_data[record])])

    arff = "".join([arff_head, arff_data])

    with open(train_path, "w") as f:
        f.write(arff)
    print(" Train set created")

print("Start time: ", start_time)
print("End time: ", time.time() - start_time)
