#! /usr/bin/bash


JSON_PATH="temp_data"
OUTPUT_PATH="temp_h5"

python3 json2vec.py --input_dir $JSON_PATH --output_dir $OUTPUT_PATH
