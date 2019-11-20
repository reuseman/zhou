from const import AFDB_DIR, AFDB_EVENTS_CSV
from pathlib import Path

import wfdb

db_dir = AFDB_DIR
db_events_csv = AFDB_EVENTS_CSV


def read_record_atr(afdb_path, record_name):
    record_path = str(Path.joinpath(afdb_path, record_name))

    return wfdb.rdann(record_path, "atr")


afdb_names = set([x.stem for x in db_dir.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")
afdb_names.remove(".directory")

af_events_records = dict()

for record_name in sorted(list(afdb_names)):
    annot_atr = read_record_atr(db_dir, record_name)
    aux_note = annot_atr.aux_note
    sample = annot_atr.sample
    print(set(aux_note))

    af_events = list()
    current_range = list()

    for i in range(0, len(aux_note)):
        if aux_note[i] == "(AFIB":
            current_range.append(sample[i])
            if i == len(aux_note) - 1:
                current_range.append(9205759)
                af_events.append(current_range.copy())
                current_range.clear()
        elif current_range:
            current_range.append(sample[i])
            af_events.append(current_range.copy())
            current_range.clear()

    af_events_records[record_name] = af_events
    print(af_events)

with open(db_events_csv, "w") as f:
    for record, events in af_events_records.items():
        f.write(record + "\n")

        for event in events:
            interval = str(event[0]) + "\t" + str(event[1]) + "\n"
            f.write(interval)

        f.write("\n")
