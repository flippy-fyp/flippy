from repro import _get_and_write_align_results, _write_overall_results
from lib.runner import Runner
from lib.args import Arguments
from lib.sharedtypes import OverallResultsT
from consts import BACH10_PATH, EXPERIMENTAL_RESULTS_PATH
from lib.eprint import eprint
import sys
import os
import re


def w_a_search():
    experimental_arg = "w_a_search"
    w_a_range = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for w_a in w_a_range:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(
            EXPERIMENTAL_RESULTS_PATH, experimental_arg, str(w_a)
        )
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with w_a: {w_a}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    "nsgt",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--backend_backtrack",
                    "--w_a",
                    str(w_a),
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
            print(f"Finished following: {piece} with w_a: {w_a}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def backend_backtrack_search():
    experimental_arg = "backend_backtrack_search"
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for backend_backtrack in [True, False]:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(
            EXPERIMENTAL_RESULTS_PATH, experimental_arg, str(backend_backtrack)
        )
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print(
                "========================================================================"
            )
            print(
                f"Starting to follow: {piece} with backend_backtrack: {backend_backtrack}"
            )
            print(
                "========================================================================"
            )
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args_str = [
                "--score_midi_path",
                score_midi_path,
                "--cqt",
                "nsgt",
                "--perf_wave_path",
                perf_wave_path,
                "--backend_output",
                output_align_path,
                "--w_a",
                "0.5",
            ]
            if backend_backtrack:
                args_str += ["--backend_backtrack"]

            args = Arguments().parse_args(args_str)

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

            print(
                "========================================================================"
            )
            print(
                f"Finished following: {piece} with backend_backtrack: {backend_backtrack}"
            )
            print(
                "========================================================================"
            )
        _write_overall_results(overall_results, output_base_dir)


def search_window_search():
    experimental_arg = "search_window_search"
    search_window_range = [
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
    ]
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for search_window in search_window_range:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(
            EXPERIMENTAL_RESULTS_PATH, experimental_arg, str(search_window)
        )
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with search_window: {search_window}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    "nsgt",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--backend_backtrack",
                    "--w_a",
                    "0.5",
                    "--search_window",
                    str(search_window),
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
            print(f"Finished following: {piece} with search_window: {search_window}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def max_run_count_search():
    experimental_arg = "max_run_count_search"
    max_run_count_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for max_run_count in max_run_count_range:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(
            EXPERIMENTAL_RESULTS_PATH, experimental_arg, str(max_run_count)
        )
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with max_run_count: {max_run_count}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    "nsgt",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--backend_backtrack",
                    "--w_a",
                    "0.5",
                    "--search_window",
                    "500",
                    "--max_run_count",
                    str(max_run_count),
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
            print(f"Finished following: {piece} with max_run_count: {max_run_count}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def hop_len_search():
    experimental_arg = "hop_len_search"
    hop_len_range = [512, 1024, 2048, 4096, 8192, 16384]
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for hop_len in hop_len_range:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(
            EXPERIMENTAL_RESULTS_PATH, experimental_arg, str(hop_len)
        )
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with hop_len: {hop_len}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(output_base_dir, piece)
            os.makedirs(output_align_dir, exist_ok=True)
            output_align_path = os.path.join(output_align_dir, "align.txt")

            args = Arguments().parse_args(
                [
                    "--score_midi_path",
                    score_midi_path,
                    "--cqt",
                    "nsgt",
                    "--perf_wave_path",
                    perf_wave_path,
                    "--backend_output",
                    output_align_path,
                    "--backend_backtrack",
                    "--w_a",
                    "0.5",
                    "--search_window",
                    "500",
                    "--hop_len",
                    str(hop_len),
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
            print(f"Finished following: {piece} with hop_len: {hop_len}")
            print("=============================================")
        _write_overall_results(overall_results, output_base_dir)


def slice_hop_ratio_search():
    pass


func_map = {
    "w_a_search": w_a_search,  # 0.5 chosen
    "backend_backtrack_search": backend_backtrack_search,  # no difference
    "search_window_search": search_window_search,  # 500 chosen
    "max_run_count_search": max_run_count_search,  # stay with 3 (no difference in [2,10])
    "hop_len_search": hop_len_search,
    "slice_hop_ratio_search": slice_hop_ratio_search,
}

if __name__ == "__main__":
    experimental_args = sys.argv[1:]
    if len(experimental_args) == 0:
        eprint("No experimental arg given--running everything!")
        for name, f in func_map.items():
            print("++++++++++++++++++++++++++++++++++++")
            print(f"Starting: {name}")
            print("++++++++++++++++++++++++++++++++++++")
            f()
            print("++++++++++++++++++++++++++++++++++++")
            print(f"Finished: {name}")
            print("++++++++++++++++++++++++++++++++++++")
        sys.exit(0)
    elif len(experimental_args) != 1:
        eprint(f"Unknown experimental arg: {experimental_args}.")
        sys.exit(1)
    experimental_arg = experimental_args[0]
    if experimental_arg in func_map:
        func_map[experimental_arg]()
    else:
        eprint(f"Unknown experimental arg: {experimental_arg}.")
        sys.exit(1)
