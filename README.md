# Flippy

Score-follower.

## Requirements
- Cloned repository with LFS
- Python 3 (Tested on Python 3.8, Ubuntu 20.04)

## Setup
(Run [`scripts/install.sh`](./scripts/install.sh) to get these automatically for Ubuntu 20.04)
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases)
- Requirements: `pip install -r requirements.txt`
- Install `nsgt` separately: `pip install nsgt`
- [Optional] Install [`fftw`](http://fftw.org/download.html)
- [Optional] Audio playback: `ffmpeg`

## Development
- Initialise pre-commit: `pre-commit install`

### WSL2 Note
- For audio playback, PulseAudio is required. See [here](https://www.linuxuprising.com/2021/03/how-to-get-sound-pulseaudio-to-work-on.html) for a guide.

## Usage
```bash
python flippy.py
```

## Results Reproduction

These scripts reproduce results shown in [project report](https://github.com/flippy-fyp/flippy-report/blob/main/main.pdf).

To run everything:
```bash
python repro.py
```

### `bwv846_feature`
```bash
python repro.py bwv846_feature
```

Plots the extracted features from the first 15 seconds of the Prelude and Fugue of Bach's BWV846 to `repro_results/bwv846_feature`.

### `bach10_feature`
```bash
python repro.py bach10_feature
```
Plots the extracted features from the first 15 seconds of all Bach10 pieces to `repro_results/bach10_feature`.
