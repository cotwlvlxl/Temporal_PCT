# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmpose import __version__
from mmpose.apis import train_model
from mmpose.utils import collect_env, get_root_logger
from mmpose.datasets import build_dataset

from models import build_posenet

from torchviz import make_dot
import torch.onnx
from tools.train import parse_args

args = parse_args()

cfg = Config.fromfile(args.config)

if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
model = build_posenet(cfg.model)

try:
    dummy_data = torch.empty(32, 3, 256, 256, dtype = torch.float32)
    make_dot(model(dummy_data), params=dict(model.named_parameters())).render("model_torchviz", format="png")
    print("Model visualization saved as model_torchviz.png")
    # torch.onnx.export(model, dummy_data, "model.onnx", export_params=True,
    #                   opset_version=11, do_constant_folding=True,
    #                   input_names=['input'], output_names=['output'],
    #                   dynamic_axes={'input': {0: 'batch_size'},
    #                                 'output': {0: 'batch_size'}})
    # print("Model exported to ONNX format successfully.")
except Exception as e:
    print(f'Error saving initial model: {e}')
    raise