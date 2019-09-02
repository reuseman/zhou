from pathlib import Path

import af_zhou as zhou
import utils

# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
generated_path = Path.joinpath(datasets_path, "generated")
generated_explicit_path = Path.joinpath(generated_path, "encoded_entropy")

afdb_path = Path.joinpath(datasets_path, "afdb")
csv_af_path = Path.joinpath(datasets_path, "af_events.csv")

read_corrected = True
entropy_threshold = 0.639


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


correct_af_events = utils.get_af_events(csv_af_path)

afdb_names = set([x.stem for x in afdb_path.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")


records_data = dict()

for record_name in afdb_names:
    # READ QRS RECORD
    annot_qrs = utils.read_record_qrs(afdb_path, record_name, read_corrected)
    # print("LEN QRS: ", len(annot_qrs.sample))

    # COMPUTE ENTROPY
    bpm_list = zhou.compute_bpm(annot_qrs.sample)
    # print("LEN BPM: ", len(bpm_list))
    sy_list = zhou.compute_sy(bpm_list)
    wv_list = zhou.compute_wv(sy_list)
    entropy = zhou.compute_entropy(wv_list)
    predictions = list(map(lambda x: 1 if x >= entropy_threshold else 0, entropy))
    # print("LEN ENTROPY: ", len(entropy))

    # COMPUTE ORACLE
    patient_af_events = correct_af_events[record_name]
    binary_af_samples = utils.compute_binary_af_samples(patient_af_events)
    oracle = utils.compute_binary_af_qrs(binary_af_samples, annot_qrs)
    # print("LEN ORACLE: ", len(oracle))

    # COMPUTE UNIMOL ONE
    unimol_entropy = compute_unimol_entropy(bpm_list)
    # print("LEN UNIMOL ENTROPY: ", len(unimol_entropy))

    records_data[record_name] = [predictions[7:], unimol_entropy, oracle[9:]]


# GENERATE FOLDERS WITH DATASET
for record_name in afdb_names:
    # Folder's name is the name of the record wich is used to test
    folder_path = Path.joinpath(generated_explicit_path, record_name)
    folder_path.mkdir(exist_ok=True)

    # Test.arff is created
    test_path = Path.joinpath(folder_path, "test.arff")
    with open(test_path, "w") as f:
        f.write("@relation test\n")
        f.write("@attribute zhou_entr numeric\n")
        f.write("@attribute unim_entr numeric\n")
        f.write("@attribute physionet {0,1}\n")
        f.write("@data\n")

        for i in range(0, len(records_data[record_name][0])):
            row = "{}, {}, {}\n".format(
                records_data[record_name][0][i],
                records_data[record_name][1][i],
                records_data[record_name][2][i],
            )
            f.write(row)

    # Train.arff is created
    train_path = Path.joinpath(folder_path, "train.arff")
    with open(train_path, "w") as f:
        f.write("@relation train\n")
        f.write("@attribute zhou_entr numeric\n")
        f.write("@attribute unim_entr numeric\n")
        f.write("@attribute physionet {0,1}\n")
        f.write("@data\n")

        for record in afdb_names:
            if record == record_name:
                continue

            for i in range(0, len(records_data[record][0])):
                row = "{}, {}, {}\n".format(
                    records_data[record][0][i],
                    records_data[record][1][i],
                    records_data[record][2][i],
                )
                f.write(row)
