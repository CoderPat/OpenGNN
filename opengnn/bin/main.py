#!/usr/bin/env/python3
import argparse
import os
from typing import Union, Dict

import tensorflow as tf

from opengnn.models import catalog
from opengnn.runner import Runner
from opengnn.config import load_model, load_config
from opengnn.utils.misc import classes_in_module


def _prefix_paths(prefix: str, paths: Union[str, Dict[str, str]])-> Union[str, Dict[str, str]]:
    """Recursively prefix paths.
    Args:
      prefix: The prefix to apply.
      data: A dict of relative paths.
    Returns:
      The updated dict.
    """
    if isinstance(paths, dict):
        for key, path in paths.items():
            paths[key] = _prefix_paths(prefix, path)
        return paths
    else:
        path = paths
        new_path = os.path.join(prefix, path)
        if os.path.isfile(new_path):
            return new_path
        else:
            return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",
                        choices=["train_and_eval", "train", "infer"],
                        help="Run type.")
    parser.add_argument("--config", required=True, nargs="+",
                        help="List of configuration files.")
    parser.add_argument("--model_type",
                        choices=list(classes_in_module(catalog)),
                        help="Model type from the catalog.")
    parser.add_argument("--model", default="",
                        help="Custom model configuration file.")
    parser.add_argument("--data_dir", default="",
                        help="If set, data files are expected to be relative "
                        "to this location.")
    parser.add_argument("--features_file",
                        help="Run inference on this file.")
    parser.add_argument("--predictions_file", default="",
                        help="File used to save predictions. If not set, "
                        "predictions are printed on the standard output.")
    args = parser.parse_args()

    # Load and merge run configurations.
    config = load_config(args.config)
    if args.data_dir:
        config["data"] = _prefix_paths(args.data_dir, config["data"])

    if not os.path.isdir(config["model_dir"]):
        tf.logging.info("Creating model directory %s", config["model_dir"])
        os.makedirs(config["model_dir"])
    model = load_model(
        config["model_dir"],
        model_file=args.model,
        model_name=args.model_type)
    session_config = tf.ConfigProto()
    runner = Runner(model, config, session_config=session_config, gpu_allow_growth=True)

    if args.run == "train_and_eval":
        runner.train_and_evaluate()
    elif args.run == "train":
        runner.train()
    elif args.run == "infer":
        if not args.features_file or not args.predictions_file:
            parser.error("--features_file and --prediction_file are"
                         " required for inference.")
        runner.infer(args.features_file, args.predictions_file)


if __name__ == "__main__":
    main()
