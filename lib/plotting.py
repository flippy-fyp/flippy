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
    figwidth: int = 6,
    figheight: int = 4,
):
    fig = plt.figure(figsize=(figwidth, figheight))
    img = librosa.display.specshow(
        cqt.T,
        sr=fs,
        x_axis="time",
        y_axis="cqt_note",
        fmin=fmin,
        hop_length=hop_length,
    )
    fig.colorbar(img)
    plt.savefig(output_path)
