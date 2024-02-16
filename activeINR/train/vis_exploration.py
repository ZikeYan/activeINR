#!/usr/bin/env python
import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
import cv2
import faulthandler

import open3d.visualization.gui as gui
from activeINR.visualisation import vis_window
from activeINR.modules import mapping
from activeINR.modules import navigation

if __name__ == "__main__":
    faulthandler.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="activeINR.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    parser.add_argument("--scene_id", default="None", help="specify test scene")
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

    # open3d vis window --------------------------------------------------------
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = vis_window.VisWindow(
        trainer,
        explorer,
        mapping.mapper,
        mono,
    )
    app.run()