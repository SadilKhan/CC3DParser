#! /usr/bin/bash

INPUT_DIR="/home/mkhan/Codes/cc3dparser/output"
FORM="h5"
SAVE_DIR="output_dir"

python3 export2step.py --input_dir $INPUT_DIR --form $FORM --save_dir $SAVE_DIR