import os
import glob
from tqdm import tqdm
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from utils.file_utils import ensure_dir,get_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="source folder")
    parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
    parser.add_argument("--save_dir",type=str,default="output_step",help="Save directory")
    parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_paths = get_files(input_dir)
    ensure_dir(args.save_dir)
    for file_path in tqdm(out_paths):
        create_model(file_path,args)

def create_model(file_path,args):
    # try:
    print(file_path)
    if args.form == "h5":
        with h5py.File(file_path, 'r') as fp:
            keyName=list(fp.keys())[0]
            out_vec = fp[keyName][:].astype(np.float64)
            out_shape = vec2CADsolid(out_vec)
    else:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        out_shape = create_CAD(cad_seq)

    # except Exception as e:
    #     print(e)
    #     #print("load and create failed.")
    #     return None    
    if args.filter:
        analyzer = BRepCheck_Analyzer(out_shape)
        if not analyzer.IsValid():
            print("detect invalid.")
            return None
    
    name = file_path.split("/")[-1].split(".")[0]
    subdir="/".join(file_path.split("/")[-3:-1]) # Only for CC3D Dataset. For Other dataset, use subdir=""
    save_path=os.path.join(args.save_dir,subdir,name+".step")
    if not os.path.exists(args.save_dir+f"/{subdir}"):
        os.makedirs(args.save_dir+f"/{subdir}")
    try:
        write_step_file(out_shape, save_path)
    except:
        print("Problem Saving the CAD Model")
        pass

if __name__=="__main__":
    main()