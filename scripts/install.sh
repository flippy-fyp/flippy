#!/bin/bash

set -eo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")" && cd ..

# Install fluidsynth
sudo apt-get install fluidsynth

# Install Python requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install nsgt

# Install fftw
wget http://fftw.org/fftw-3.3.9.tar.gz
tar -xzf fftw-3.3.9.tar.gz
rm -f fftw-3.3.9.tar.gz
cd fftw-3.3.9
./configure
sudo make
sudo make install
make check
# go back to project root
cd "$(dirname "$0")" && cd ..

# Install ffmpeg
sudo apt-get install ffmpeg
