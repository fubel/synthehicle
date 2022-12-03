import argparse
import os
import pathlib
import warnings

import numpy as np
import json
from tqdm import tqdm

from .valid_scenes import VALID_TEST

parser = argparse.ArgumentParser(
    description="MOTChallenge to Synthehicle JSON converter."
)


parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=True,
    help="Root directory of dataset, e.g., data/synthehicle/",
)
parser.add_argument(
    "-c",
    "--cameras",
    type=str,
    required=False,
    help="Path to camera config file., e.g., splits/test.txt",
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    required=False,
    help="Pattern to prediction files., e.g., gt/gt.txt",
    default="gt/gt.txt",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to output json file, e.g., predictions.json",
)

args = parser.parse_args()

if args.cameras is not None:
    with open(args.cameras, "r") as file:
        cameras = file.read().splitlines()
else:
    cameras = VALID_TEST

output = {}

for cam in tqdm(cameras, desc="Iterating camera paths"):
    if cam not in VALID_TEST:
        warnings.warn(
            f"Camera {cam} is not a valid synthehicle test split camera and will be ignored in evaluation."
        )
    scene_name, cam_name = pathlib.Path(cam).parts
    file = os.path.join(args.data_dir, cam, args.pattern)
    if os.path.isfile(file):
        data = np.loadtxt(file, delimiter=",").astype(np.int64).tolist()
    else:
        raise FileNotFoundError(f"File does not exist: {file}")
    if scene_name in output.keys():
        if cam_name not in output[scene_name].keys():
            output[scene_name][cam_name] = data
        else:
            raise ValueError(f"Duplicate scene and camera: {scene_name}, {cam_name}")
    else:
        output[scene_name] = {}
        output[scene_name][cam_name] = data

with open(args.output, "w") as f:
    json.dump(output, f)
