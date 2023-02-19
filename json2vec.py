import os
import open3d as o3d
import argparse
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import h5py
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.macro import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",help="Input directory for json files")
    parser.add_argument("--output_dir",help="Output directory for h5 files")
    args=parser.parse_args()

    #JSON_PATH=os.path.join(args.input_dir+"*/*/*/*")
    JSON_PATH=os.path.join(args.input_dir+"/*")
    SAVE_DIR=args.output_dir

    allJsonFiles = glob(JSON_PATH)
    for js in tqdm(allJsonFiles):
        process_one(js,SAVE_DIR)


def process_one(json_path,save_dir):
    json_id=json_path.split('/')[-1].split('.')[0]
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    cad_seq = CADSequence.from_dict(data)
    sample_points =cad_seq.sample_points()
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(sample_points)

    # TASK: Normalize 
    cad_seq.normalize()
    cad_seq.numericalize()
    cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, None, pad=False)

    # WORK ON SUBDIR
    subdir=""
    json_save_path = os.path.join(save_dir,subdir, json_id + ".h5")
    pc_path= os.path.join(save_dir,subdir, json_id + ".ply")
    truck_dir = os.path.dirname(json_save_path)
    pc_dir=os.path.dirname(pc_path)

    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    if not os.path.exists(pc_dir):
        os.makedirs(pc_dir)

    o3d.io.write_point_cloud(pc_path,pcd)
    with h5py.File(json_save_path, 'w') as fp:
        fp.create_dataset("vec", data=cad_vec, dtype=np.int32)


if __name__ == '__main__':
    main()
