import argparse
import json
from .eval_tools import evaluate_synthehicle_json

parser = argparse.ArgumentParser(description="Synthehicle local evaluator.")

parser.add_argument(
    "-p",
    "--prediction",
    type=str,
    required=True,
    help="Path to prediction json.",
)

parser.add_argument(
    "-g",
    "--ground-truth",
    type=str,
    required=True,
    help="Path to ground truth json.",
)

args = parser.parse_args()

with open(args.prediction) as f:
    p = json.load(f)

with open(args.ground_truth) as f:
    g = json.load(f)

print(evaluate_synthehicle_json(p, g))
