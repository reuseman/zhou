from const import AFDB_DIR
from pathlib import Path

import utils
import wfdb

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

read_corrected = True

afdb_names = set([x.stem for x in AFDB_DIR.iterdir() if x.is_file()])
afdb_names.remove(".gitkeep")
afdb_names.remove("03665")
afdb_names.remove("00735")

for record_name in afdb_names:
    print(Path.joinpath(AFDB_DIR, record_name))
    record_path = Path.joinpath(AFDB_DIR, record_name)
    record = wfdb.rdrecord(record_path, channel_names=["ECG1"])
    annot_qrs = utils.read_record_qrs(AFDB_DIR, record_name, read_corrected)

    signal = record.p_signal
    signal = signal.squeeze()
    avg = np.average(signal)
    print("NUMPY AVG:", avg)

    linear_detrend = scipy.signal.detrend(signal, type="linear")
    constant_detrend = scipy.signal.detrend(signal, type="constant")[0:annot_qrs.sample[0]]

    signal_fft = np.fft.fft(signal).real
    linear_fft = np.fft.fft(linear_detrend).real
    constant_fft = np.fft.fft(constant_detrend, n=16).real

    fig, axes = plt.subplots(3, 2, sharex=True)

    axes[0, 0].plot(signal)
    axes[0, 0].set_title("Original Signal")

    axes[0, 1].plot(signal_fft)
    axes[0, 1].set_title("Fourier on signal")
    axes[0, 1].set_xlabel("frequency")

    axes[1, 0].plot(linear_detrend)
    axes[1, 0].set_title("Signal detrend Linear scipy")
    axes[1, 0].set_xlabel("time/sample")

    axes[1, 1].plot(linear_fft)
    axes[1, 1].set_title("Fourier on Linear scipy")
    axes[1, 1].set_xlabel("frequency")

    axes[2, 0].plot(constant_detrend)
    axes[2, 0].set_title("Signal detrend Constant scipy")
    axes[2, 0].set_xlabel("time/sample")

    axes[2, 1].plot(constant_fft)
    axes[2, 1].set_title("Signal detrend Constant scipy")
    axes[2, 1].set_xlabel("frequency")

    plt.tight_layout()
    plt.show()

    break
