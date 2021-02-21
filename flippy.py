from tap import Tap
from utils.eprint import eprint
from sys import exit
from os import path
class ArgumentParser(Tap):
    mode: str = "online" # Mode: `offline` or `online`.
    dtw: str = "oltw" # DTW Algo: `classical` or `oltw`. `classical` is only available for the `offline` mode.
    cqt: str = "slicq" # CQT Algo: `slicq` or `librosa`. `librosa` is only available for the `offline` mode.
    max_run_count: int = 3 # `MaxRunCount` for `online` mode with `oltw` DTW.
    search_window: int = 250 # `SearchWindow` for `online` mode with `oltw` DTW.
    fmin: float = 130.8 # Minimum frequency (Hz) for CQT.
    fmax: float = 4186.0 # Maximum frequency (Hz) for CQT.
    slice_len: int = 2048 # Slice length for `slicq` cqt.
    transition_len: int = 0 # Transition length for `slicq` cqt.
    hop: int = 2048 # Hop length for `librosa` cqt.

    perf_wave_path: str # Path to performance WAVE file.
    score_midi_path: str # Path to score MIDI.

def sanitize_arguments(args: ArgumentParser):
    def eprint_and_exit(msg: str):
        eprint(f"Argument Error: {msg}.")
        eprint("Use the `--help` flag to show the help message.")
        exit(1)

    if args.mode not in ["online", "offline"]:
        eprint_and_exit(f"Unknown mode: `{args.mode}`")

    if args.dtw not in ["oltw", "classical"]:
        eprint_and_exit(f"Unknown dtw: `{args.dtw}`")

    if args.cqt not in ["slicq", "librosa"]:
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

    if args.transition_len < 0:
        eprint_and_exit(f"transition_len must be positive")

    if args.hop < 0:
        eprint_and_exit(f"hop must be positive")

    if args.mode == "online":
        if args.dtw != "oltw":
            eprint_and_exit("For `online` mode only `oltw` dtw is accepted")
        if args.cqt != "slicq":
            eprint_and_exit("For `online` mode only `slicq` cqt is accepted")

    if not path.isfile(args.perf_wave_path):
        eprint_and_exit(f"Performance WAVE file ({args.perf_wave_path}) does not exist")
        
    if not path.isfile(args.score_midi_path):
        eprint_and_exit(f"Score MIDI file ({args.perf_midi_path}) does not exist")

if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    sanitize_arguments(args)
    