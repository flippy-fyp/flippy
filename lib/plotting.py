from lib.constants import DEFAULT_SAMPLE_RATE
import matplotlib.pyplot as plt  # type: ignore
import librosa  # type: ignore
import librosa.display  # type: ignore
import numpy as np


def plot_cqt_to_file(
    output_path: str,
    cqt: np.ndarray,
    fmin: float,
    hop_length: int = 2048,
    fs: int = DEFAULT_SAMPLE_RATE,
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

    plt.savefig(output_path)
