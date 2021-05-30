#!/bin/bash

set -eo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")" && cd ../..

if [ "$#" -ne 4 ]; then
    echo "Need four parameters: piece group, piece name, udphost, udpport"
    exit 1
fi

piece_group=$1
piece_name=$2
udphost=$3
udpport=$4

echo "Piece Group: $piece_group"
echo "Piece Name: $piece_name"
echo "UDP Host: $udphost"
echo "UDP Port: $udpport"

perf_wave_path="data/qual/$piece_group/$piece_name/$piece_name.wav"
score_pickle_path="data/qual/$piece_group/$piece_name/$piece_name.pickle"

echo "perf_wave_path: $perf_wave_path"
echo "score_pickle_path: $score_pickle_path"

./scripts/qual/run_pickle.sh "$perf_wave_path" "$score_pickle_path" "$udphost" "$udpport"
