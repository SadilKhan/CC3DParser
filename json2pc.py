import os
import glob
import json
import numpy as np
import random
import h5py
from tqdm import tqdm
from trimesh.sample import sample_surface
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD
from utils.pc_utils import write_ply, read_ply
from utils.file_utils import get_files

FAILED_PLY=0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir",help="Input Directory for json/h5 file",required=True)
    parser.add_argument("-o","--output_dir",help="Output Directory",default="output_cad")
    parser.add_argument("-n","--num_point",help="Number of points for the point clouds",default=8096)
    args=parser.parse_args()

    H5_DIR=args.input_dir
    SAVE_DIR=args.output_dir
    N_POINTS=args.num_point

    all_h5files=get_files(H5_DIR)
    for file in tqdm(all_h5files):
        process_one(file,args)
    print(f"Failed CAD Model:{FAILED_PLY*100/len(all_h5files)}")

def process_one(data_path,args):
    global FAILED_PLY
    data_id=data_path.split("/")[-1].split(".")[0]

    with open(data_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        FAILED_PLY+=1
        print("create_CAD failed:", data_id)
        return None
    try:
        out_pc = CADsolid2pc(shape, args.num_point, data_id.split("/")[-1])
    except Exception as e:
        FAILED_PLY+=1
        print("convert point cloud failed:", data_id)
        return None

    save_path = os.path.join(args.save_dir, data_id + ".ply")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    write_ply(out_pc, save_path)


if __name__=="__main__":
    main()