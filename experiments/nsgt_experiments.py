from lib.constants import DEFAULT_SAMPLE_RATE
from experiments.librosa_experiments import plot_cqt
from lib.cqt.cqt_nsgt import (
    extract_features_nsgt_cqt,
    get_nsgt_params,
    extract_features_nsgt_slicq,
    get_slicq_engine,
)
import librosa  # type: ignore
import librosa.display  # type: ignore
import numpy as np
import time


def plot_nsgt_features():
    audio, _ = librosa.load(
        "./tmp/wtk1-prelude1.wav", sr=DEFAULT_SAMPLE_RATE, duration=10, mono=True
    )

    fmin, fmax = get_nsgt_params()
    cqt = extract_features_nsgt_cqt(audio, fmin, fmax)

    # print(len(cqt[0]))

    plot_cqt(cqt, fmin, hop_length=2048 // 100)  # 100 is the default hop len in nsgt


def plot_nsgt_features_slice():
    sl_tr_ratio = 4
    hop_length = 2048
    frame_length = hop_length * sl_tr_ratio

    audio_stream = librosa.stream(
        "./tmp/wtk1-prelude1.wav",
        1,
        frame_length,
        hop_length,
        mono=True,
        fill_value=0,
        duration=10,
    )

    fmin, fmax = get_nsgt_params()
    slicq = get_slicq_engine(frame_length, sl_tr_ratio, fmin, fmax)

    n_bins = 59  # Based on the defaults
    prev_slice: np.ndarray = np.zeros(n_bins)
    cqt = []

    start_time = time.time()
    for audio_slice in audio_stream:
        cqt_slice = extract_features_nsgt_slicq(slicq, sl_tr_ratio, audio_slice)
        # save for tmp
        tmp_slice = cqt_slice
        # get diff
        cqt_slice = cqt_slice - prev_slice
        # clip
        cqt_slice = cqt_slice.clip(0)  # type: ignore
        # update in cqt
        cqt.append(cqt_slice)
        # update prev
        prev_slice = tmp_slice
    # convert to ndarray
    cqt = np.array(cqt)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    plot_cqt(cqt, fmin, hop_length=hop_length)
