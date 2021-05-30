from lib.mputils import write_list_to_queue
from lib.components.follower import Follower
import time
import statistics
from flippy_quantitative_testbench.utils.match import (
    MatchResult,
    safe_div,
)
from lib.runner import Runner
from lib.audio import cut_wave
from lib.cqt.cqt_nsgt import get_nsgt_params
from lib.plotting import plot_cqt_to_file
from lib.sharedtypes import (
    AlignResultsT,
    ExtractedFeature,
    ExtractedFeatureQueue,
    OverallResultsT,
)
from lib.components.synthesiser import Synthesiser
from lib.constants import DEFAULT_SAMPLE_RATE
from lib.components.audiopreprocessor import AudioPreprocessor
from lib.eprint import eprint
from lib.args import Arguments
from consts import (
    BACH10_PATCH_PATH,
    BACH10_PATH,
    BWV846_PATH,
    MISALIGN_THRESHOLD_MS_RANGE,
    REPRO_RESULTS_PATH,
)
import os
import multiprocessing as mp
from typing import Any, Dict, List, Tuple, TypeVar, TypedDict, Union
import numpy as np
import sys
import re
from flippy_quantitative_testbench.utils.bench import bench
import json

MisalignThresT = int


class AggregatedResult(TypedDict):
    # piecewise: mean across pieces
    piecewise_precision_rate: float
    total_precision_rate: float
    piecewise_results: MatchResult


AggregatedResultsT = Dict[MisalignThresT, AggregatedResult]


def _dict_mean(dict_list: List[Dict[str, Union[int, float]]]) -> Dict[str, float]:
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def _write_overall_results(overall_results: OverallResultsT, output_base_dir: str):
    total_dict: AggregatedResultsT = {}
    for thres, results in overall_results.items():
        precision_rates = list(map(lambda x: x["precision_rate"], results))
        piecewise_precision_rate = sum(precision_rates) / len(precision_rates)

        total_num = sum(map(lambda x: x["total_num"], results))
        miss_num = sum(map(lambda x: x["miss_num"], results))
        misalign_num = sum(map(lambda x: x["misalign_num"], results))

        align_num = total_num - miss_num - misalign_num
        total_precision_rate = safe_div(float(align_num), total_num)
        piecewise_results = _dict_mean(results)

        total_dict[thres] = {
            "piecewise_precision_rate": piecewise_precision_rate,
            "total_precision_rate": total_precision_rate,
            "piecewise_results": piecewise_results,
        }

    total_output_path = os.path.join(output_base_dir, "results.json")
    with open(total_output_path, "w+") as f:
        total_dict_str = json.dumps(total_dict, indent=4)
        f.write(total_dict_str)


def _get_and_write_align_results(
    output_align_path: str, ref_align_path: str, align_results_path: str
) -> AlignResultsT:
    align_results = {
        thres: bench(output_align_path, ref_align_path, thres)
        for thres in MISALIGN_THRESHOLD_MS_RANGE
    }
    with open(align_results_path, "w+") as f:
        align_result_str = json.dumps(align_results, indent=4)
        f.write(align_result_str)
    return align_results


def _read_overall_results(overall_results_path: str) -> AggregatedResultsT:
    f = open(overall_results_path)
    t = f.read().strip()
    f.close()
    x = json.loads(t)
    return x


"""
def cqt_frame_time():
    repro_arg = "cqt_frame_time"
    piece_path = os.path.join(BACH10_PATH, "01-AchGottundHerr", "01-AchGottundHerr.wav")
    fmin, fmax = get_nsgt_params()
    output_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg)
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict[int, float]] = {}
    for cqt in ["nsgt", "librosa_pseudo", "librosa_hybrid", "librosa"]:
        print("=============================================")
        print(f"Running with cqt: {cqt}")
        print("=============================================")
        results[cqt] = {}
        for frame_len in [8192, 16384, 32768, 65536]:
            print(f"=== FRAME_LEN: {frame_len} ===")
            q = mp.Queue()
            ap = AudioPreprocessor(
                DEFAULT_SAMPLE_RATE,
                2048,
                frame_len,
                piece_path,
                False,
                0.0005,
                "online",
                cqt,
                fmin,
                fmax,
                q,
                False,
                10,
            )
            start_time = time.perf_counter()
            ap_proc = mp.Process(target=ap.start)
            ap_proc.start()
            while True:
                x = q.get()
                if x is None:
                    break
            ap_proc.join()
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            print(f"=== TIME TAKEN: {time_taken}s ===")
            results[cqt][frame_len] = time_taken
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w+") as f:
        results_str = json.dumps(results, indent=4)
        f.write(results_str)
        import matplotlib.pyplot as plt

    name_mapping = {
        "librosa": "CQT",
        "nsgt": "NSGT-CQT",
        "librosa_pseudo": "CQT (Pseudo)",
        "librosa_hybrid": "CQT (Hybrid)",
    }

    data: Dict[str, List[Tuple[int, float]]] = {
        name_mapping[cqt]: [(x, y) for x, y in res.items()]
        for cqt, res in results.items()
    }

    plt.figure()
    for name, scores in data.items():
        [x, y] = zip(*scores)
        plt.plot(x, y)
    plt.ylabel("Time Taken (s)")
    plt.xlabel("Frame Length")
    plt.legend(data.keys(), loc="upper left")
    plt.tight_layout()
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, "output.pdf"))
"""


def cqt_time():
    repro_arg = "cqt_time"
    piece_path = os.path.join(BACH10_PATH, "01-AchGottundHerr", "01-AchGottundHerr.wav")
    fmin, fmax = get_nsgt_params()
    output_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg)
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict[int, float]] = {}
    for cqt in ["nsgt", "librosa_pseudo"]:
        print("=============================================")
        print(f"Running with cqt: {cqt}")
        print("=============================================")
        results[cqt] = {}
        for piece_len_s in range(10, 70, 10):
            print(f"=== DURATION: {piece_len_s} ===")
            q = mp.Queue()
            ap = AudioPreprocessor(
                DEFAULT_SAMPLE_RATE,
                2048,
                8192,
                piece_path,
                False,
                0.0005,
                "online",
                cqt,
                fmin,
                fmax,
                q,
                False,
                float(piece_len_s),
            )
            start_time = time.perf_counter()
            ap_proc = mp.Process(target=ap.start)
            ap_proc.start()
            while True:
                x = q.get()
                if x is None:
                    break
            ap_proc.join()
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            print(f"=== TIME TAKEN: {time_taken}s ===")
            results[cqt][piece_len_s] = time_taken
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w+") as f:
        results_str = json.dumps(results, indent=4)
        f.write(results_str)
        import matplotlib.pyplot as plt

    name_mapping = {
        "nsgt": "NSGT-CQT",
        "librosa_pseudo": "CQT (Pseudo)",
    }

    data: Dict[str, List[Tuple[int, float]]] = {
        name_mapping[cqt]: [(x, y) for x, y in res.items()]
        for cqt, res in results.items()
    }

    data["Real-time boundary"] = [(x, float(x)) for x in range(10, 70, 10)]

    plt.figure()
    for name, scores in data.items():
        [x, y] = zip(*scores)
        plt.plot(x, y)
    plt.ylabel("Time Taken (s)")
    plt.xlabel("Audio Length (s)")
    plt.legend(data.keys(), loc="upper left")
    plt.tight_layout()
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, "output.pdf"))


def dtw_time():
    repro_arg = "dtw_time"
    np.random.seed(42)
    output_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg)
    os.makedirs(output_dir, exist_ok=True)
    full_results: Dict[str, Dict[int, List[float]]] = {}
    seq_len_range = range(100, 1001, 100)
    for seq_len in seq_len_range:
        for i in range(1, 4):
            print("=============================================")
            print(f"({i}/3): Running with seq_len: {seq_len}")
            print("=============================================")
            S = np.random.rand(seq_len, 50)
            P = np.random.rand(seq_len, 50)
            for dtw in ["classical", "oltw"]:
                print(f"=== DTW: {dtw} ===")
                output_queue = mp.Queue()
                P_queue = mp.Queue()
                write_list_to_queue(P, P_queue)
                follower = Follower(
                    "offline", dtw, 500, 3, output_queue, P_queue, S, 1.0, 1.0, 1.0
                )

                start_time = time.perf_counter()
                follower_proc = mp.Process(target=follower.start)
                follower_proc.start()
                while True:
                    x = output_queue.get()
                    if x is None:
                        break
                follower_proc.join()

                end_time = time.perf_counter()
                time_taken = end_time - start_time
                print(f"=== TIME TAKEN: {time_taken}s ===")

                if not dtw in full_results:
                    full_results[dtw] = {}
                if not seq_len in full_results[dtw]:
                    full_results[dtw][seq_len] = []
                full_results[dtw][seq_len].append(time_taken)
    averaged_results: Dict[str, Dict[int, float]] = {
        dtw: {seq_len: statistics.mean(times) for seq_len, times in res.items()}
        for dtw, res in full_results.items()
    }
    full_results_path = os.path.join(output_dir, "full_results.json")
    with open(full_results_path, "w+") as f:
        results_str = json.dumps(full_results, indent=4)
        f.write(results_str)

    averaged_results_path = os.path.join(output_dir, "averaged_results.json")
    with open(averaged_results_path, "w+") as f:
        results_str = json.dumps(averaged_results, indent=4)
        f.write(results_str)

    import matplotlib.pyplot as plt

    name_mapping = {
        "oltw": "OLTW",
        "classical": "Classical DTW",
    }

    data: Dict[str, List[Tuple[int, float]]] = {
        name_mapping[cqt]: [(x, y) for x, y in res.items()]
        for cqt, res in averaged_results.items()
    }

    # data["Real-time boundary"] = [(x, float(x)) for x in range(10, 70, 10)]

    plt.figure()
    for name, scores in data.items():
        [x, y] = zip(*scores)
        plt.plot(x, y)
    plt.ylabel("Time Taken (s)")
    plt.xlabel("Sequence Length")
    plt.legend(data.keys(), loc="upper left")
    plt.tight_layout()
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, "output.pdf"))


def bach10_feature():
    repro_arg = "bach10_feature"
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for cqt in ["nsgt", "librosa_pseudo"]:
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
                8192,
                perf_wave_path,
                False,
                0.0005,
                "online",
                cqt,
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
                REPRO_RESULTS_PATH, repro_arg, cqt, piece_basename
            )
            os.makedirs(output_plot_dir, exist_ok=True)
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
            print(
                f"Finished plotting features for {piece_basename} to {output_plot_path}"
            )


def bwv846_feature():
    repro_arg = "bwv846_feature"
    pieces = ["prelude", "fugue"]
    for cqt in ["nsgt", "librosa_pseudo"]:
        for piece in pieces:
            fmin, fmax = get_nsgt_params()
            score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
            score_wave_path = Synthesiser(
                score_midi_path, DEFAULT_SAMPLE_RATE
            ).synthesise(15)
            print(f"Synthesised to: {score_wave_path}")
            output_queue: ExtractedFeatureQueue = mp.Queue()
            ap = AudioPreprocessor(
                DEFAULT_SAMPLE_RATE,
                2048,
                8192,
                score_wave_path,
                False,
                0.0005,
                "online",
                cqt,
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
            output_plot_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt, piece)
            os.makedirs(output_plot_dir, exist_ok=True)
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
    repro_arg = "bwv846_align"
    pieces = ["prelude", "fugue"]
    cqts = ["nsgt", "librosa"]
    for cqt in cqts:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt)
        os.makedirs(output_base_dir, exist_ok=True)
        for piece in pieces:
            print("=============================================")
            print(f"Starting to align: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
            perf_wave_path = os.path.join(BWV846_PATH, piece, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
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
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                ]
            )

            runner = Runner(args)
            runner.start()

            ref_align_path = os.path.join(BWV846_PATH, piece, f"{piece}.align.txt")
            align_results_path = os.path.join(output_align_dir, "result.json")
            align_results = _get_and_write_align_results(
                output_align_path, ref_align_path, align_results_path
            )

            for thres, align_result in align_results.items():
                if thres not in overall_results:
                    overall_results[thres] = []
                overall_results[thres].append(align_result)

            print("=============================================")
            print(f"Finished aligning: {piece} with cqt: {cqt}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def bwv846_follow():
    repro_arg = "bwv846_follow"
    pieces = ["prelude", "fugue"]
    cqts = ["nsgt", "librosa_pseudo"]
    for cqt in cqts:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt)
        os.makedirs(output_base_dir, exist_ok=True)
        for piece in pieces:
            print("=============================================")
            print(f"Starting to follow: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(BWV846_PATH, piece, f"{piece}.r.mid")
            perf_wave_path = os.path.join(BWV846_PATH, piece, f"{piece}.wav")

            output_align_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    cqt,
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--simulate_performance",
                    "--backend_backtrack",
                    "--w_a",
                    "0.5",
                ]
            )

            runner = Runner(args)
            runner.start()

            ref_align_path = os.path.join(BWV846_PATH, piece, f"{piece}.align.txt")
            align_results_path = os.path.join(output_align_dir, "result.json")
            align_results = _get_and_write_align_results(
                output_align_path, ref_align_path, align_results_path
            )

            for thres, align_result in align_results.items():
                if thres not in overall_results:
                    overall_results[thres] = []
                overall_results[thres].append(align_result)

            print("=============================================")
            print(f"Finished following: {piece} with cqt: {cqt}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def bach10_align():
    repro_arg = "bach10_align"
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    cqts = ["nsgt", "librosa"]
    for cqt in cqts:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt)
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to align: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(BACH10_PATCH_PATH, piece, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt, piece)
            os.makedirs(output_align_dir, exist_ok=True)
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

            ref_align_path = os.path.join(BACH10_PATH, piece, f"{piece}.txt")
            align_results_path = os.path.join(output_align_dir, "result.json")
            align_results = _get_and_write_align_results(
                output_align_path, ref_align_path, align_results_path
            )

            for thres, align_result in align_results.items():
                if thres not in overall_results:
                    overall_results[thres] = []
                overall_results[thres].append(align_result)

            print("=============================================")
            print(f"Finished aligning: {piece} with cqt: {cqt}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def bach10_follow():
    repro_arg = "bach10_follow"
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    cqts = ["nsgt", "librosa_pseudo"]
    for cqt in cqts:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt)
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with cqt: {cqt}")
            print("=============================================")
            score_midi_path = os.path.join(BACH10_PATCH_PATH, piece, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg, cqt, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    cqt,
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--simulate_performance",
                    "--backend_backtrack",
                    "--w_a",
                    "0.5",
                ]
            )

            runner = Runner(args)
            runner.start()

            ref_align_path = os.path.join(BACH10_PATH, piece, f"{piece}.txt")
            align_results_path = os.path.join(output_align_dir, "result.json")
            align_results = _get_and_write_align_results(
                output_align_path, ref_align_path, align_results_path
            )

            for thres, align_result in align_results.items():
                if thres not in overall_results:
                    overall_results[thres] = []
                overall_results[thres].append(align_result)

            print("=============================================")
            print(f"Finished following: {piece} with cqt: {cqt}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def _plot_precision_plot(data: Dict[str, List[Tuple[int, float]]], output_dir: str):
    import matplotlib.pyplot as plt

    plt.figure()
    for name, scores in data.items():
        [x, y] = zip(*scores)
        plt.plot(x, y)
    plt.ylabel("Total Precision Rate")
    plt.xlabel("Misalign Threshold (ms)")
    plt.legend(data.keys(), loc="lower right")
    plt.tight_layout()
    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-")

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, "output.pdf"))


def _plot_precision_wrapper(repro_dict: Dict[str, Dict[str, str]], output_dir: str):
    # health check to make sure all required folders exist
    for repro_dict_arg, cqts in repro_dict.items():
        for cqt in cqts:
            dir_to_check = os.path.join(REPRO_RESULTS_PATH, repro_dict_arg, cqt)
            if not os.path.exists(dir_to_check):
                raise ValueError(
                    f"Please run repro for {repro_dict_arg} before running this step!"
                )

    # now read in all the OverallResults
    # map from name to OverallResultsT
    overall_results: Dict[str, AggregatedResultsT] = {}
    for repro_dict_arg, cqts in repro_dict.items():
        for cqt, name in cqts.items():
            overall_results_path = os.path.join(
                REPRO_RESULTS_PATH, repro_dict_arg, cqt, "results.json"
            )
            res = _read_overall_results(overall_results_path)
            overall_results[name] = res

    # we're interested in total precision only
    data = {
        name: [(int(thres), x["total_precision_rate"]) for thres, x in results.items()]
        for name, results in overall_results.items()
    }
    _plot_precision_plot(data, output_dir)


def bach10_plot_precision():
    repro_arg = "bach10_plot_precision"
    output_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg)
    os.makedirs(output_dir, exist_ok=True)
    repro_dict = {
        "bach10_align": {
            "librosa": "CQT Offline",
            "nsgt": "NSGT-CQT Offline",
        },
        "bach10_follow": {
            "librosa_pseudo": "CQT (Pseudo) Online",
            "nsgt": "NSGT-CQT Online",
        },
    }
    _plot_precision_wrapper(repro_dict, output_dir)


def bwv846_plot_precision():
    repro_arg = "bwv846_plot_precision"
    output_dir = os.path.join(REPRO_RESULTS_PATH, repro_arg)
    os.makedirs(output_dir, exist_ok=True)
    repro_dict = {
        "bwv846_align": {
            "librosa": "CQT Offline",
            "nsgt": "NSGT-CQT Offline",
        },
        "bwv846_follow": {
            "librosa_pseudo": "CQT (Pseudo) Online",
            "nsgt": "NSGT-CQT Online",
        },
    }
    _plot_precision_wrapper(repro_dict, output_dir)


"""
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
    os.makedirs(output_plot_dir, exist_ok=True)
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
"""

func_map = {
    # "cqt_frame_time": cqt_frame_time,
    "cqt_time": cqt_time,
    "dtw_time": dtw_time,
    "bwv846_feature": bwv846_feature,
    "bach10_feature": bach10_feature,
    "bwv846_align": bwv846_align,
    "bach10_align": bach10_align,
    "bwv846_follow": bwv846_follow,
    "bach10_follow": bach10_follow,
    "bach10_plot_precision": bach10_plot_precision,
    "bwv846_plot_precision": bwv846_plot_precision,
}

if __name__ == "__main__":
    repro_args = sys.argv[1:]
    if len(repro_args) == 0:
        eprint("No repro arg given--running everything!")
        for name, f in func_map.items():
            print("++++++++++++++++++++++++++++++++++++")
            print(f"Starting: {name}")
            print("++++++++++++++++++++++++++++++++++++")
            f()
            print("++++++++++++++++++++++++++++++++++++")
            print(f"Finished: {name}")
            print("++++++++++++++++++++++++++++++++++++")
        sys.exit(0)
    elif len(repro_args) != 1:
        eprint(f"Unknown repro args: {repro_args}. Please see README.md")
        sys.exit(1)
    repro_arg = repro_args[0]
    if repro_arg in func_map:
        func_map[repro_arg]()
    else:
        eprint(f"Unknown repro arg: {repro_arg}. Please see README.md")
        sys.exit(1)
