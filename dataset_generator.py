from pathlib import Path
from scipy.signal import detrend
from spectrum import aryule
from const import AFDB_DIR, GENERATED_DIR, AF_EVENTS_CSV, ENTROPY_THRESHOLD

import numpy as np
import wfdb

import zhou
import utils


# Settings
read_corrected = True
add_fft = True
add_ar = True
fft_n = 16
ar_n = 4


# Paths
GENERATED_DIR.mkdir(exist_ok=True)
folder_to_create = GENERATED_DIR / "explicit_entropy_fft_16_ar_4"
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


def get_arff_head(relation_name, add_fft=False, add_ar=False):
    arff_head = "@relation {}\n".format(relation_name)
    arff_head = "".join(
        [
            arff_head,
            "@attribute zhou_explicit_entr numeric\n@attribute unim_entr numeric\n",
        ]
    )

    if add_fft:
        for i in range(1, fft_n + 1):
            arff_head = "".join([arff_head, "@attribute fft_{} numeric\n".format(i)])

    if add_ar:
        for i in range(1, ar_n + 1):
            f.write("@attribute ar_{} numeric\n".format(i))
            arff_head = "".join([arff_head, "@attribute ar_{} numeric\n".format(i)])

    return "".join([arff_head, "@attribute physionet {0,1}\n@data\n"])


def get_arff_data(record):
    data = ""

    for i in range(0, len(record[0])):
        # Output of str(list[record[2][i]]) is "[1,2,3,..,fft_n]"
        #   with the range [1:-1] the squared parenthesis are removed
        fft_temp = str(list(record[2][i]))[1:-1]
        ar_temp = str(list(record[3][i]))[1:-1]

        "".join(
            [
                data,
                "{}, {}, {}, {}, {}\n".format(
                    record[0][i], record[1][i], fft_temp, ar_temp, record[4][i]
                )
            ]
        )

    return data


correct_af_events = utils.get_af_events(AF_EVENTS_CSV)

afdb_names = set([x.stem for x in AFDB_DIR.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")
if add_fft or add_ar:
    afdb_names.remove("03665")
    afdb_names.remove("00735")

records_data = dict()

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
    #   numpy arrays of shape (fft_n,) and/or (ar_n) are added
    #    to the fft_values and/or ar_values
    if add_fft or add_ar:
        fft_values = list()
        ar_values = list()
        
        signal = read_heart_signal(record_name)
        constant_detrend = detrend(signal, type="constant")
        
        for i in range(0, len(annot_qrs.sample) - 1):
            rr_signal = constant_detrend[annot_qrs.sample[i] : annot_qrs.sample[i + 1]]

            if add_fft:
                fft_values.append(np.fft.fft(rr_signal, fft_n).real)

            if add_ar:
                ar_values.append(aryule(rr_signal, ar_n)[0])

        print("LEN FFT: ", len(fft_values))
        print("LEN AR: ", len(ar_values))

    # COMPUTE ORACLE
    patient_af_events = correct_af_events[record_name]
    binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
    oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)
    print("LEN ORACLE: ", len(oracle))

    records_data[record_name] = [
            entr[7:],
            unimol_entropy,
    ]
    if add_fft:
        records_data[record_name].append(fft_values[9:])
    if add_ar:
        records_data[record_name].append(ar_values[9:])
    records_data[record_name].append(oracle[9:])

    """ print("\nLENGTH IN THE RECORDS DATA: ")
    print("ENTR: ", len(records_data[record_name][0]))
    print("UNIM: ", len(records_data[record_name][1]))
    print("FFT_: ", len(records_data[record_name][2]))
    print("AR_V: ", len(records_data[record_name][3]))
    print("ORAC: ", len(records_data[record_name][-1])) """

# GENERATE FOLDERS WITH DATASET
for record_name in afdb_names:
    # Folder's name is the name of the record wich is used to test
    sub_folder_record = Path.joinpath(folder_to_create, record_name)
    sub_folder_record.mkdir(exist_ok=True)

    # Create Test.arff
    test_path = Path.joinpath(sub_folder_record, "test.arff")
    with open(test_path, "w") as f:
        f.write(get_arff_head("test", add_fft))
        f.write(get_arff_data(records_data[record_name]))
        x = get_arff_data(records_data[record_name])

    # Create Train.arff
    train_path = Path.joinpath(sub_folder_record, "train.arff")
    with open(train_path, "w") as f:
        f.write(get_arff_head("train", add_fft))

        for record in afdb_names:
            if record == record_name:
                continue

            f.write(get_arff_data(records_data[record]))
