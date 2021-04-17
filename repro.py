from lib.runner import Runner
from lib.audio import cut_wave
from lib.cqt.cqt_nsgt import get_nsgt_params
from lib.plotting import plot_cqt_to_file
from lib.sharedtypes import ExtractedFeature, ExtractedFeatureQueue
from lib.components.synthesiser import Synthesiser
from lib.constants import DEFAULT_SAMPLE_RATE
from lib.components.audiopreprocessor import AudioPreprocessor
from lib.eprint import eprint
from lib.args import Arguments
from consts import BACH10_PATH, BWV846_PATH, REPRO_RESULTS_PATH
import os
import multiprocessing as mp
from typing import List
import numpy as np
import sys
import re


def bach10_feature():
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for piece_path in bach10_piece_paths:
        piece_basename = os.path.basename(piece_path)
        print(f"Processing {piece_basename}")
        fmin, fmax = get_nsgt_params()
        full_perf_wave_path = os.path.join(piece_path, f"{piece_basename}.wav")
        perf_wave_path = cut_wave(0, 15, full_perf_wave_path)
        output_queue: ExtractedFeatureQueue = mp.Queue()
        ap = AudioPreprocessor(
            DEFAULT_SAMPLE_RATE,
            2048,
            4,
            perf_wave_path,
            False,
            "online",
            "nsgt",
            fmin,
            fmax,
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
        print(f"Features for {piece_basename} extracted successfully")
        # convert to ndarray
        features_ndarray = np.array(features)
        output_plot_dir = os.path.join(
            REPRO_RESULTS_PATH, "bach10_feature", piece_basename
        )
        if not os.path.exists(output_plot_dir):
            os.makedirs(output_plot_dir)
        output_plot_path = os.path.join(output_plot_dir, "features.pdf")
        print(f"Plotting features for {piece_basename} to {output_plot_path}")
        plot_cqt_to_file(
            output_plot_path,
            features_ndarray,
            fmin,
            2048,
            DEFAULT_SAMPLE_RATE,
            8,
            5,
        )
        print(f"Finished plotting features for {piece_basename} to {output_plot_path}")


def bwv846_feature():
    pieces = ["prelude", "fugue"]
    for piece in pieces:
        fmin, fmax = get_nsgt_params()
        score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
        score_wave_path = Synthesiser(score_midi_path, DEFAULT_SAMPLE_RATE).synthesise(
            15
        )
        print(f"Synthesised to: {score_wave_path}")
        output_queue: ExtractedFeatureQueue = mp.Queue()
        ap = AudioPreprocessor(
            DEFAULT_SAMPLE_RATE,
            2048,
            4,
            score_wave_path,
            False,
            "online",
            "nsgt",
            fmin,
            fmax,
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
        if not os.path.exists(output_plot_dir):
            os.makedirs(output_plot_dir)
        output_plot_path = os.path.join(output_plot_dir, "features.pdf")
        print(f"Plotting features for {piece} to {output_plot_path}")
        plot_cqt_to_file(
            output_plot_path,
            features_ndarray,
            fmin,
            2048,
            DEFAULT_SAMPLE_RATE,
            8,
            5,
        )
        print(f"Finished plotting features for {piece} to {output_plot_path}")


def bwv846_align():
    pieces = ["prelude", "fugue"]
    cqts = ["nsgt", "librosa"]
    for piece in pieces:
        for cqt in cqts:
            print("=============================================")
            print(f"Starting to align: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
            perf_wave_path = os.path.join(
                BWV846_PATH, piece, f"{piece}.mp3"
            )  # mp3 is fine

            output_align_dir = os.path.join(
                REPRO_RESULTS_PATH, "bwv846_align", cqt, piece
            )
            if not os.path.exists(output_align_dir):
                os.makedirs(output_align_dir)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--mode",
                    "offline",
                    "--score_midi_path",
                    score_midi_path,
                    "--dtw",
                    "classical",
                    "--cqt",
                    cqt,
                    "--hop_len",
                    "2048",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                ]
            )

            runner = Runner(args)
            runner.start()

            print("=============================================")
            print(f"Finished aligning: {piece} with cqt: {cqt}")
            print("=============================================")


def bach10_align():
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    cqts = ["nsgt", "librosa"]
    for piece_path in bach10_piece_paths:
        piece = os.path.basename(piece_path)
        for cqt in cqts:
            print("=============================================")
            print(f"Starting to align: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(
                REPRO_RESULTS_PATH, "bach10_align", cqt, piece
            )
            if not os.path.exists(output_align_dir):
                os.makedirs(output_align_dir)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--mode",
                    "offline",
                    "--score_midi_path",
                    score_midi_path,
                    "--dtw",
                    "classical",
                    "--cqt",
                    cqt,
                    "--hop_len",
                    "2048",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                ]
            )

            runner = Runner(args)
            runner.start()

            print("=============================================")
            print(f"Finished aligning: {piece} with cqt: {cqt}")
            print("=============================================")


def playground():
    fmin, fmax = get_nsgt_params()
    full_perf_wave_path = os.path.join("./tmp/wtk1-prelude1-performance.wav")
    perf_wave_path = cut_wave(0, 15, full_perf_wave_path)
    output_queue: ExtractedFeatureQueue = mp.Queue()
    ap = AudioPreprocessor(
        DEFAULT_SAMPLE_RATE,
        2048,
        4,
        perf_wave_path,
        False,
        "online",
        "nsgt",
        fmin,
        fmax,
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
    # convert to ndarray
    features_ndarray = np.array(features)
    output_plot_dir = os.path.join(
        REPRO_RESULTS_PATH, "playground", "wtk-prelude1-performance"
    )
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
    output_plot_path = os.path.join(output_plot_dir, "features.pdf")
    print(f"Plotting features for wtk-prelude1-performance to {output_plot_path}")
    plot_cqt_to_file(
        output_plot_path,
        features_ndarray,
        fmin,
        2048,
        DEFAULT_SAMPLE_RATE,
        8,
        5,
    )
    print(
        f"Finished plotting features for wtk-prelude1-performance to {output_plot_path}"
    )


if __name__ == "__main__":
    repro_args = sys.argv[1:]
    if len(repro_args) != 1:
        eprint(f"Unknown repro args: {repro_args}. Please see README.md")
        sys.exit(1)
    repro_arg = repro_args[0]

    if repro_arg == "bwv846_feature":
        bwv846_feature()
    elif repro_arg == "bach10_feature":
        bach10_feature()
    elif repro_arg == "bwv846_align":
        bwv846_align()
    elif repro_arg == "bach10_align":
        bach10_align()
    elif repro_arg == "playground":
        playground()
    else:
        eprint(f"Unknown repro arg: {repro_arg}. Please see README.md")
        sys.exit(1)
