from lib.plotting import plot_cqt_to_file
from lib.sharedtypes import ExtractedFeature
from lib.components.synthesiser import Synthesiser
from lib.constants import DEFAULT_SAMPLE_RATE
from lib.components.audiopreprocessor import AudioPreprocessor
from lib.eprint import eprint
from consts import BWV846_PATH, REPRO_RESULTS_PATH
from lib.utils import quantise_hz_midi
import os
import multiprocessing as mp
from typing import Optional, List
import numpy as np
import sys


def bwv846_feature():
    pieces = ["prelude", "fugue"]
    for piece in pieces:
        fmin = quantise_hz_midi(130.8)
        fmax = quantise_hz_midi(4186.0)
        score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
        score_wave_path = Synthesiser(score_midi_path, DEFAULT_SAMPLE_RATE).synthesise(
            10
        )
        output_queue: "mp.Queue[Optional[ExtractedFeature]]" = mp.Queue()
        ap = AudioPreprocessor(
            DEFAULT_SAMPLE_RATE,
            score_wave_path,
            2048,
            2048 * 4,
            False,
            "online",
            "nsgt",
            fmin,
            fmax,
            2048,
            4,
            output_queue,
        )
        ap_proc = mp.Process(target=ap.start)
        ap_proc.start()
        features: List[ExtractedFeature] = []
        while True:
            feat = output_queue.get()
            if feat is None:
                break
            features.append(feat)
        ap_proc.join()
        print(f"Features for {piece} extracted successfully")
        # convert to ndarray
        features_ndarray = np.array(features)
        output_plot_dir = os.path.join(REPRO_RESULTS_PATH, "bwv846_feature", piece)
        os.makedirs(output_plot_dir)
        output_plot_path = os.path.join(output_plot_dir, "features.pdf")
        print(f"Plotting features for {piece} to {output_plot_path}")
        plot_cqt_to_file(
            output_plot_path, features_ndarray, fmin, 2048, DEFAULT_SAMPLE_RATE
        )
        print(f"Finished plotting features for {piece} to {output_plot_path}")


if __name__ == "__main__":
    repro_args = sys.argv[1:]
    if len(repro_args) != 1:
        eprint(f"Unknown repro args: {repro_args}. Please see README.md")
        sys.exit(1)
    repro_arg = repro_args[0]

    if repro_arg == "bwv846_feature":
        bwv846_feature()
    else:
        eprint(f"Unknown repro arg: {repro_arg}. Please see README.md")
        sys.exit(1)
