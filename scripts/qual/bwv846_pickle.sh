#!/bin/bash

set -eo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")" && cd ../..

if [ "$#" -ne 3 ]; then
    echo "Need three parameters: piece name, udphost, udpport"
    exit 1
fi

piece=$1
udphost=$2
udpport=$3

echo "Piece: $piece"
echo "UDP Host: $udphost"
echo "UDP Port: $udpport"

perf_wave_path="data/bwv846/$piece/$piece.wav"
score_pickle_path="data/bwv846/$piece/$piece.pickle"

echo "perf_wave_path: $perf_wave_path"
echo "score_pickle_path: $score_pickle_path"

./scripts/qual/run_pickle.sh "$perf_wave_path" "$score_pickle_path" "$udphost" "$udpport"
