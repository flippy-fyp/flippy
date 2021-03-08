from lib.sharedtypes import ModeType, DTWType, CQTType, BackendType
from lib.utils import quantise_hz_midi
from tap import Tap  # type: ignore
from lib.eprint import eprint
from sys import exit
from os import path


class ArgumentParser(Tap):
    mode: ModeType = "online"  # Mode: `offline` or `online`.
    dtw: DTWType = "oltw"  # DTW Algo: `classical` or `oltw`. `classical` is only available for the `offline` mode.
    cqt: CQTType = "nsgt"  # CQT Algo: `nsgt`, `librosa_pseudo`, `librosa_hybrid` or `librosa`. `librosa` is only available for the `offline` mode.
    max_run_count: int = 3  # `MaxRunCount` for `online` mode with `oltw` DTW.
    search_window: int = 250  # `SearchWindow` for `online` mode with `oltw` DTW.
    fmin: float = 130.8  # Minimum frequency (Hz) for CQT.
    fmax: float = 4186.0  # Maximum frequency (Hz) for CQT.
    slice_len: int = (
        2048  # Slice length for `nsgt` cqt, or hop_length in `librosa` cqt.
    )
    transition_slice_ratio: int = 4  # Transition to slice length ratio for `nsgt` cqt.

    perf_wave_path: str  # Path to performance WAVE file.
    score_midi_path: str  # Path to score MIDI.

    backend: BackendType = (
        "alignment"  # Alignment result type: `alignment` or `timestamp`.
    )


def sanitize_arguments(args: ArgumentParser) -> ArgumentParser:
    def eprint_and_exit(msg: str):
        eprint(f"Argument Error: {msg}.")
        eprint("Use the `--help` flag to show the help message.")
        exit(1)

    if args.mode not in ["online", "offline"]:
        eprint_and_exit(f"Unknown mode: `{args.mode}`")

    if args.dtw not in ["oltw", "classical"]:
        eprint_and_exit(f"Unknown dtw: `{args.dtw}`")

    if args.cqt not in ["nsgt", "librosa"]:
        eprint_and_exit(f"Unknown cqt: `{args.cqt}`")

    if args.max_run_count < 0:
        eprint_and_exit(f"max_run_count must be positive")

    if args.search_window < 0:
        eprint_and_exit(f"search_window must be positive")

    if args.fmin < 0:
        eprint_and_exit(f"fmin must be positive")

    if args.fmax < 0:
        eprint_and_exit(f"fmax must be positive")

    if args.fmax <= args.fmin:
        eprint_and_exit(f"fmax > fmin not fulfilled")

    if args.slice_len < 0:
        eprint_and_exit(f"slice_len must be positive")

    if args.transition_slice_ratio < 0:
        eprint_and_exit(f"transition_slice_ratio must be positive")

    if args.mode == "online":
        if args.dtw != "oltw":
            eprint_and_exit("For `online` mode only `oltw` dtw is accepted")
        if args.cqt not in ("nsgt", "librosa_pseudo", "librosa_hybrid"):
            eprint_and_exit(
                "For `online` mode only `nsgt`, `librosa_pseudo` or `librosa_hybrid` cqt is accepted"
            )

    if not path.isfile(args.perf_wave_path):
        eprint_and_exit(f"Performance WAVE file ({args.perf_wave_path}) does not exist")

    if not path.isfile(args.score_midi_path):
        eprint_and_exit(f"Score MIDI file ({args.score_midi_path}) does not exist")

    if args.slice_len % 100 != 0 and args.mode == "offline" and args.cqt == "nsgt":
        eprint_and_exit(
            f"slice_len ({args.slice_len}) for offline nsgt must be a multiple of 100"
        )

    if args.backend not in ("alignment", "timestamp"):
        eprint_and_exit("backend must be one of `alignment` or `timestamp`")

    # ---- MUTATIVE ----

    # quantize fmin and fmax
    fmin = quantise_hz_midi(args.fmin)
    eprint(f"fmin quantised from {args.fmin} to {fmin}")
    args.fmin = fmin

    fmax = quantise_hz_midi(args.fmax)
    eprint(f"fmax quantised from {args.fmax} to {fmax}")
    args.fmax = fmax

    return args


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    args = sanitize_arguments(args)
