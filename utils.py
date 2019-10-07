from pathlib import Path
from math import sqrt

import csv
import wfdb

from const import RECORD_LEN


# Gives back Annotations-QRS
def read_record_qrs(afdb_path, record_name, read_corrected=False):
    corrected_records = ["05091", "07859"]
    record_path = str(Path.joinpath(afdb_path, record_name))

    if read_corrected and record_name in corrected_records:
        annot_qrs = wfdb.rdann(record_path, "qrsc")  # , sampto=800000)
    else:
        annot_qrs = wfdb.rdann(record_path, "qrs")  # , sampto=800000)

    return annot_qrs


# Read af_evenets.csv and create a dict containing as key:str the patient,
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


# Generates an array of samples matching the record, where 1 is AF and 0 not AF
def compute_binary_af_samples(patient_af_events):
    binary_af_samples = [0] * RECORD_LEN

    for af_records in patient_af_events:
        binary_af_samples[af_records[0] : af_records[1] + 1] = map(
            lambda x: 1, binary_af_samples[af_records[0] : af_records[1] + 1]
        )

    return binary_af_samples


# Generates an array of qrs, where 1 is an heart beat with AF and 0 not AF
def compute_binary_af_qrs(binary_af_samples, annot_qrs):
    binary_af_qrs = list()

    start_value = annot_qrs.sample[0]
    for i in range(1, len(annot_qrs.sample)):
        end_value = annot_qrs.sample[i] + 1

        ones = binary_af_samples[start_value:end_value].count(1)

        interval_length = end_value - start_value
        percentage_of_ones = ones / interval_length

        # CHANGE THIS PART TO COMPUTE IN DIFFERENT WAY THE ORACLE
        element = 1 if percentage_of_ones > 0.5 else 0
        binary_af_qrs.append(element)

        start_value = end_value + 1

    return binary_af_qrs


# ---------------
# METRICS METHODS
# ---------------


def compute_metrics(tp, tn, fp, fn):
    # Sensitivity
    try:
        se = tp / (tp + fn)
    except ZeroDivisionError:
        se = "NaN"

    # Specificity
    try:
        sp = tn / (tn + fp)
    except ZeroDivisionError:
        sp = "NaN"

    # Precision or Positive predictive value
    try:
        ppv = tp / (tp + fp)
    except ZeroDivisionError:
        ppv = "NaN"

    # Accuracy
    try:
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        acc = "NaN"

    return se, sp, ppv, acc


def compute_mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    try:
        return num / den
    except ZeroDivisionError:
        return "NaN"
