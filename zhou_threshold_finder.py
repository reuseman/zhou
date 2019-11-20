from const import LOCAL_PREDICTION_RESULTS_AFDB
import matplotlib.pyplot as plt

import numpy as np
import csv
import pandas as pd

record_tresholds = set(
    [x.stem for x in LOCAL_PREDICTION_RESULTS_AFDB.iterdir() if x.is_file()]
)

#record_tresholds.remove("best_results_tresholds")

best_results = dict()
for record in record_tresholds:
    path = LOCAL_PREDICTION_RESULTS_AFDB / (record + ".csv")
    print(record)

    # THRESHOLD	TP	TN	FP	FN	SE	SP	PPV	ACC
    df = pd.read_csv(path)
    df["SP"] = 1 - df["SP"]

    perfect_point = np.array((0, 1))
    min_dist = 100
    threshold_index = 0

    for index, row in df.iterrows():
        if np.isnan(row["SP"]):
            row["SP"] = 0
        current_point = np.array((row["SP"], row["SE"]))
        dist = np.linalg.norm(current_point - perfect_point)

        if dist <= min_dist:
            min_dist = dist
            threshold_index = index

    print("The optimal threshold is: ", df["THRESHOLD"][threshold_index])
    print(
        f"Where the SP is {1 - df['SP'][threshold_index]} and the SE is {df['SE'][threshold_index]}\n"
    )

    best_results[record] = df.iloc[threshold_index]
    best_results[record]['SP'] = 1 - best_results[record]['SP'] 


best_results_list = []
for key, value in best_results.items():
    temp = value.tolist()
    temp.insert(0, str(key))
    best_results_list.append(temp)

best_results_csv = LOCAL_PREDICTION_RESULTS_AFDB / "best_results_tresholds.csv"
with open(best_results_csv, mode="w") as results_file:
    fieldnames = [
        "RECORD_NAME",
        "THRESHOLD",
        "TP",
        "TN",
        "FP",
        "FN",
        "SE",
        "SP",
        "PPV",
        "ACC",
    ]

    results_writer = csv.writer(results_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)

    results_writer.writerow(fieldnames)
    results_writer.writerows(best_results_list)

