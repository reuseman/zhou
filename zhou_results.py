from const import PICKLE_RESULT, ZHOU_RESULTS_CSV

import pickle as pkl
import csv

import utils


def get_classifications(results):
    tp, tn, fp, fn = 0, 0, 0, 0

    for result in results:
        tp += result[1]
        tn += result[2]
        fp += result[3]
        fn += result[4]

    return tp, tn, fp, fn


with open(PICKLE_RESULT, "rb") as f:
    results = pkl.load(f)

tp, tn, fp, fn = get_classifications(results)
se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)

print("-----------------------------------")
print("TRUE POSITIVE : ", tp)
print("TRUE NEGATIVE : ", tn)
print("FALSE POSITIVE: ", fp)
print("FALSE NEGATIVE: ", fn)
print("-----------------------------------")
print("SENSITIVITY   : ", se)
print("SPECIFICITY   : ", sp)
print("PRECISION     : ", ppv)
print("ACCURACY      : ", acc)
print("-----------------------------------")

# Save results on a .csv
with open(ZHOU_RESULTS_CSV, mode="w") as results_file:
    fieldnames = ["RECORD", "TP", "TN", "FP", "FN", "LEN(oracle)"]

    results_writer = csv.writer(
        results_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )

    results_writer.writerow(fieldnames)
    for result in results:
        results_writer.writerow(result)
