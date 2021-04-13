# Flippy

Score-follower.

## Requirements
- Python 3 (Tested on Python 3.8, Ubuntu 20.04)
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
