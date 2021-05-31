#!/bin/bash

set -eo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")" && cd ../..

if [ "$#" -ne 4 ]; then
    echo "Need three parameters: perf_wave_path, score_midi_path, udphost, udpport"
    exit 1
fi

perf_wave_path=$1
score_midi_path=$2
udphost=$3
udpport=$4

set -x
echo "STARTING..."
python ./flippy.py \
    --perf_wave_path "$perf_wave_path" \
    --score_midi_path "$score_midi_path" \
    --mode online \
    --backend timestamp \
    --backend_output "udp:$udphost:$udpport" \
    --play_performance_audio \
    --simulate_performance \
    --w_a 0.5 \
    --sleep_compensation 0.0008 \
    2>&1 | tee -a /dev/stderr
echo "FINISHED"
