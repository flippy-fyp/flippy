from lib.constants import DEFAULT_SAMPLE_RATE
from lib.midi import midi_to_audio
from lib.cqt.cqt_librosa import (
    extract_features_librosa_cqt,
    extract_features_librosa_pseudo_cqt,
    extract_features_librosa_hybrid_cqt,
    extract_slice_features_librosa_pseudo_cqt,
    extract_slice_features_librosa_hybrid_cqt,
    get_librosa_params,
)
import matplotlib.pyplot as plt  # type: ignore
import librosa  # type: ignore
import librosa.display  # type: ignore
import soundfile as sf  # type: ignore
import numpy as np
import time
import pyfftw  # type: ignore
import time

librosa.set_fftlib(pyfftw.interfaces.numpy_fft)


def conv_to_wav():
    audio = midi_to_audio("./tmp/wtk1-prelude1.mid", DEFAULT_SAMPLE_RATE)
    sf.write("./tmp/wtk-prelude1.wav", audio, samplerate=DEFAULT_SAMPLE_RATE)


def plot_cqt(
    cqt: np.ndarray, fmin: float, hop_length: int = 2048, fs: int = DEFAULT_SAMPLE_RATE
):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(
        cqt.T,
        sr=fs,
        x_axis="time",
        y_axis="cqt_note",
        ax=ax,
        fmin=fmin,
        hop_length=hop_length,
    )
    fig.colorbar(img, ax=ax)

    curr_time_int = int(time.time())
    print(curr_time_int)
    plt.savefig(f"./tmp/{curr_time_int}.pdf")


def plot_librosa_features(extractor=extract_features_librosa_cqt):
    audio, _ = librosa.load(
        "./tmp/wtk-prelude1.wav", sr=DEFAULT_SAMPLE_RATE, duration=10, mono=True
    )

    fmin, n_bins = get_librosa_params()
    cqt = extractor(audio, fmin, n_bins)

    plot_cqt(cqt, fmin)


def plot_librosa_features_stream(extractor=extract_slice_features_librosa_pseudo_cqt):
    frame_length = 2048
    audio_stream = librosa.stream(
        "./tmp/wtk-prelude1.wav",
        1,
        frame_length,
        frame_length,
        mono=True,
        fill_value=0,
        duration=10,
    )

    fmin, n_bins = get_librosa_params()
    prev_slice: np.ndarray = np.zeros(n_bins)
    cqt = []

    start_time = time.time()
    for audio_slice in audio_stream:
        cqt_slice = extractor(audio_slice, fmin, n_bins, DEFAULT_SAMPLE_RATE)
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
    plot_cqt(cqt, fmin)
