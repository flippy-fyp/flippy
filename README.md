# Flippy

Score-follower.

## Requirements
- Cloned repository with all submodules
```bash
git clone <REPO_URL> --recurse-submodules
```
- Python 3 (Tested on Python 3.8, Ubuntu 20.04)

## Setup
(Run [`scripts/install.sh`](./scripts/install.sh) to get these automatically for Ubuntu 20.04)
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases)
- Requirements: `pip install -r requirements.txt`
- Install `nsgt` separately: `pip install nsgt`
- Initialise pre-commit: `pre-commit install`
- [Optional] Install [`fftw`](http://fftw.org/download.html)
- [Optional] Audio playback: `ffmpeg`

### WSL2 Note
- For audio playback, PulseAudio is required. See [here](https://www.linuxuprising.com/2021/03/how-to-get-sound-pulseaudio-to-work-on.html) for a guide.

## Usage
```bash
python flippy.py --help
```

### Using the [Quantitative Testbench](https://github.com/flippy-fyp/flippy-quantitative-testbench)
- The Quantitative Testbench is already included as a submodule in this repository in [`flippy_quantitative_testbench`](./flippy_quantitative_testbench)
- The [Results Reproduction](#results-reproduction) section below use the testbench, see [`repro.py`](./repro.py) for references to the testbench reprository.
- To output compatible score-follower output (also compatible with the MIREX format), set the backend type to `alignment` (default), an example is:
```bash
python flippy.py \
    --perf_wave_path <PERFORMANCE_WAVE_PATH> \ # path to the wave file of the performance
    --score_midi_path <SCORE_MIDI_PATH> \      # path to the midi score file
    --mode offline \                           # offline (alignment mode)
    --backend alignment                        # output alignment in the backend
```


### Using the [Qualitative Testbench](https://github.com/flippy-fyp/flippy-qualitative-testbench)

#### Running your own pieces
- The Qualitative Testbench can be found [here](https://github.com/flippy-fyp/flippy-qualitative-testbench)
- You need to set up a UDP Port number in the testbench--see instructions in [that repository](https://github.com/flippy-fyp/flippy-qualitative-testbench)
- With the host name and UDP Port number of the testbench machine, run flippy on `online` mode and `timestamp` backend, an example that also plays the performance audio on the score-follower machine is:
```bash
python flippy.py \
    --perf_wave_path <PERFORMANCE_WAVE_PATH> \ # path to the wave file of the performance
    --score_midi_path <SCORE_MIDI_PATH> \      # path to the midi score file
    --mode online \                            # online (following mode)
    --backend timestamp \                      # output timestamps in the backend
    --backend_output udp:<HOSTNAME>:<PORT> \   # output to stderr and the UDP server at <HOSTNAME>:<PORT>
    --play_performance_audio \                 # play the performance audio on the machine where this command is run
    --simulate_performance                     # stream the performance wave audio slices "live" into the system
```

#### Demos
See Demos subsection below.

## Demos

Demo videos are provided in the [`demos`](./demos) directory. To understand the structure and reproduce these, see the Demos Reproduction subsection below.

## Results Reproduction

### Report Results

These scripts reproduce results shown in the [project report](https://arxiv.org/abs/2205.03247).

To run everything:
```bash
python repro.py
```

#### `cqt_time`
```bash
python repro.py cqt_time
```

Plots the time taken to extract CQT featuers on different lengths of audio using the `librosa`, `nsgt` and `librosa_pseudo` and `librosa_hybrid` techniques.

#### `dtw_time`
```bash
python repro.py dtw_time
```

Plots the time taken to align sequences of different lengths using the `oltw` and `classical` DTW methods.

#### `bwv846_feature`
```bash
python repro.py bwv846_feature
```

Plots the extracted features from the first 15 seconds of the Prelude and Fugue of Bach's BWV846 to `repro_results/bwv846_feature`.

#### `bach10_feature`
```bash
python repro.py bach10_feature
```

Plots the extracted features from the first 15 seconds of all Bach10 pieces to `repro_results/bach10_feature`.

#### `bwv846_align`
```bash
python repro.py bwv846_align
```

Aligns (offline) BWV846 and then runs the testbench to output results in `repro_results/bwv846_align`.

#### `bach10_align`
```bash
python repro.py bach10_align
```

Aligns (offline) Bach10 and then runs the testbench to output results in `repro_results/bwv846_align`.

#### `bach10_follow`
```bash
python repro.py bach10_follow
```

Follows (online) Bach10 and then runs the testbench to output results in `repro_results/bach10_follow`.

#### `bwv846_follow`
```bash
python repro.py bwv846_follow
```

Follows (online) BWV846 and then runs the testbench to output results in `repro_results/bwv846_follow`.

#### `bach10_plot_precision`
```bash
python repro.py bach10_plot_precision
```

Plots total precision results for Bach10--requires `bach10_align` and `bach10_follow` repro steps to be run a priori.

#### `bwv846_plot_precision`
```bash
python repro.py bwv846_plot_precision
```

Plots total precision results for Bach10--requires `bwv846_align` and `bwv846_follow` repro steps to be run a priori.

### Demos Reproduction

Possible combinations of `<GROUP_ID>` and `<PIECE_ID>` are defined in the [QualScofo dataset](https://github.com/flippy-fyp/QualScofo).

#### [`demos`](./demos) structure

This directory contains videos of the following in action (using the qualitative testbench to visualise the following).

```bash
demos
|---videos
    |---<GROUP_ID>
        |---<PIECE_ID>.mkv
```

#### Reproduction scripts

```bash
./scripts/qual/qual.sh <GROUP_ID> <PIECE_ID> <QUALITATIVE_TESTBENCH_IP> <QUALITATIVE_TESTBENCH_PORT>
```

You may try to use the preprocessed pickle files, which should work on Python 3.8.x systems:
```bash
./scripts/qual/qual_pickle.sh <GROUP_ID> <PIECE_ID> <QUALITATIVE_TESTBENCH_IP> <QUALITATIVE_TESTBENCH_PORT>
```

## Contributing

* File bugs and/or feature requests in the [GitHub repository](https://github.com/flippy/flippy)
* Pull requests are welcome in the [GitHub repository](https://github.com/flippy/flippy)
* Buy me a Coffee ☕️ via [PayPal](https://paypal.me/lhl2617)

## Citing

### BibTeX
```
@misc{https://doi.org/10.48550/arxiv.2205.03247,
  doi = {10.48550/ARXIV.2205.03247},
  url = {https://arxiv.org/abs/2205.03247},
  author = {Lee, Lin Hao},
  keywords = {Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Musical Score Following and Audio Alignment},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
