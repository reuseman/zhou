import csv
import pickle as pkl


# This will read af_evenets.csv and create a dict containing as key:str the patient,
#   while as value:int the records of AF
def get_af_events(path):
    correct_af_interval = dict()
    new_patient = True

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for row in csv_reader:
            if row:
                if new_patient:
                    new_patient = False
                    current_patient = row[0]
                    correct_af_interval[current_patient] = []
                else:
                    correct_af_interval[current_patient].append(
                        [int(row[0]), int(row[1])]
                    )
            else:
                new_patient = True

    return correct_af_interval


def compute_metrics(tp, tn, fp, fn):
    # Sensitivity
    se = tp / (tp + fn)

    # Specificity
    sp = tn / (tn + fp)

    # Precision or Positive predictive value
    ppv = tp / (tp + fp)

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)

    return se, sp, ppv, acc


def compute_classifications(results):
    tp, tn, fp, fn = 0, 0, 0, 0

    for result in results:
        tp += result[1]
        tn += result[2]
        fp += result[3]
        fn += result[4]

    return tp, tn, fp, fn


def save_results_csv(path, results):
    with open(path, mode="w") as results_file:
        fieldnames = ["RECORD", "TP", "TN", "FP", "FN", "LEN(oracle)"]

        results_writer = csv.writer(
            results_file, delimiter=",",
            quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )

        results_writer.writerow(fieldnames)
        for result in results:
            results_writer.writerow(result)


def save_object(path, object_to_save):
    with open(path, "wb") as f:
        pkl.dump(object_to_save, f)


def read_object(path):
    with open(path, "rb") as f:
        array = pkl.load(f)
    return array
