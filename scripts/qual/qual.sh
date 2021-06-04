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

perf_wave_path="data/QualScofo/$piece_group/$piece_name/$piece_name.wav"
score_midi_path="data/QualScofo/$piece_group/$piece_name/$piece_name.mid"

echo "perf_wave_path: $perf_wave_path"
echo "score_midi_path: $score_midi_path"

./scripts/qual/run.sh "$perf_wave_path" "$score_midi_path" "$udphost" "$udpport"
