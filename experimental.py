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
    w_a_range = range(0.1, 2.1, 0.1)
    bach10_piece_paths = [
        f.path
        for f in os.scandir(BACH10_PATH)
        if f.is_dir() and bool(re.search(r"^[0-9]{2}-\w+$", os.path.basename(f.path)))
    ]
    for w_a in w_a_range:
        overall_results: OverallResultsT = {}
        output_base_dir = os.path.join(EXPERIMENTAL_RESULTS_PATH, experimental_arg, w_a)
        os.makedirs(output_base_dir, exist_ok=True)
        for piece_path in bach10_piece_paths:
            piece = os.path.basename(piece_path)
            print("=============================================")
            print(f"Starting to follow: {piece} with w_a: {w_a}")
            print("=============================================")
            score_midi_path = os.path.join(piece_path, f"{piece}.mid")
            perf_wave_path = os.path.join(piece_path, f"{piece}.wav")

            output_align_dir = os.path.join(
                EXPERIMENTAL_RESULTS_PATH, experimental_arg, w_a, piece
            )
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
                    # "--simulate_performance",
                    "--backend_backtrack",
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


func_map = {
    "w_a_search": w_a_search,
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
