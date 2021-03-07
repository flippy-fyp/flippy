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
    audio = midi_to_audio("./tmp/wtk1-prelude1.mid", 44100)
    sf.write("./tmp/wtk-prelude1.wav", audio, samplerate=44100)


def plot_cqt(cqt: np.ndarray, fmin: float, hop_length: int = 2048, fs: int = 44100):
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
    audio, _ = librosa.load("./tmp/wtk-prelude1.wav", sr=44100, duration=10, mono=True)

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
    prev_slice: np.ndarray = np.zeros((1, n_bins))
    cqt = np.empty((0, n_bins), dtype=np.float32)

    start_time = time.time()
    for audio_slice in audio_stream:
        cqt_slice = extractor(audio_slice, fmin, n_bins, 44100)
        # save for tmp
        tmp_slice = cqt_slice
        # get diff
        cqt_slice = cqt_slice - prev_slice
        # clip
        cqt_slice = cqt_slice.clip(0)  # type: ignore
        # update in cqt
        cqt = np.append(cqt, cqt_slice, axis=0)
        # update prev
        prev_slice = tmp_slice

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    plot_cqt(cqt, fmin)
