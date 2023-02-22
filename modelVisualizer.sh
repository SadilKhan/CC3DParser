#! /usr/bin/bash

INPUT_DIR="temp_h5"
FORM="h5"
SAVE_DIR="output_step"

python3 export2step.py --input_dir $INPUT_DIR --form $FORM --save_dir $SAVE_DIR