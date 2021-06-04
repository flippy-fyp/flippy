import os


REPO_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(REPO_ROOT, "data")
BWV846_PATH = os.path.join(DATA_PATH, "bwv846")
REPRO_RESULTS_PATH = os.path.join(REPO_ROOT, "repro_results")
EXPERIMENTAL_RESULTS_PATH = os.path.join(REPO_ROOT, "experimental_results")
BACH10_PATH = os.path.join(DATA_PATH, "bach10", "Bach10_v1.1")

MISALIGN_THRESHOLD_MS_RANGE = range(50, 3050, 50)
