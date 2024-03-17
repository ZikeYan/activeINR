#!/usr/bin/env python
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import argparse
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from activeINR.eval import eval_window
from activeINR.modules import mapping
from activeINR.modules import navigation

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="activeINR.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    parser.add_argument("--scene_id", default="None", help="specify test scene")
    parser.add_argument("--file", default="None", help="recorded actions")
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args
    config_file = args.config

    if (args.scene_id == "None"):
        scene_id = None
    else:
        scene_id = args.scene_id
    # init nav-------------------------------------------------------------
    explorer = navigation.Explorer(device, config_file, scene_id)
    # init trainer-------------------------------------------------------------
    trainer = mapping.Trainer(
        device,
        config_file,
        incremental="sim",
        scene_id=scene_id
        )

    w = eval_window.EvaWindow(
        trainer,
        explorer,
        mapping.mapper,
        args.file
    )