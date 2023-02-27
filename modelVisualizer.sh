#! /usr/bin/bash

INPUT_DIR="/data/3d_cluster/CC3D-OP-SEQ/cc3d_v3.0_recs_extrude"
#INPUT_DIR="test_data"
FORM="json"
SAVE_DIR="output_step"

python3 export2step.py --input_dir $INPUT_DIR --form $FORM --save_dir $SAVE_DIR