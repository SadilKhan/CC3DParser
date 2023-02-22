#! /usr/bin/bash


JSON_PATH="/data/3d_cluster/CC3D-OP-SEQ/cc3d_v3.0_recs_extrude"
OUTPUT_PATH="output"

python3 json2vec.py --input_dir $JSON_PATH --output_dir $OUTPUT_PATH
