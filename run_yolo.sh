#!/bin/bash
taskset -c 4-7 ./build/run_yolo

# Run chmod +x run_yolo.sh to be able to run ../run_yolo.sh