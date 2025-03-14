#!/usr/bin/env python3
""" Model Benchmark Script
An inference and train step benchmark script for timm models.
Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from benchmark import *
from collections import OrderedDict
from contextlib import suppress
from functools import partial

from timm.models import create_model, is_model, list_models
from timm.optim import create_optimizer_v2
from timm.data import resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

import models

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis
    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    has_fvcore_profiling = False


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


def _parse_args():
    args = argparse.Namespace()
    args.model_list = ''  # txt file based list of model names to benchmark
    args.bench = 'both'  # Benchmark mode. One of 'inference', 'train', 'both'. Defaults to 'both'
    args.detail = False  # Provide train fwd/bwd/opt breakdown detail if True. Defaults to False
    args.results_file = ''  # Output csv file for validation results (summary)
    args.num_warm_iter = 10  # Number of warmup iterations (default: 10)
    args.num_bench_iter = 40  # Number of benchmark iterations (default: 40)
    args.model = 'resnet50'  # Model architecture (default: resnet50)
    args.batch_size = 256  # Mini-batch size (default: 256)
    args.img_size = None  # Input image dimension, uses model default if empty
    args.input_size = None  # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    args.use_train_size = False  # Run inference at train size, not test-input-size if it exists
    args.num_classes = None  # Number of classes in dataset
    args.gp = None  # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None
    args.channels_last = False  # Use channels_last memory layout
    args.amp = False  # Use PyTorch Native AMP for mixed precision training. Overrides --precision arg
    args.precision = 'float32'  # Numeric precision. One of (amp, float32, float16, bfloat16, tf32)
    args.torchscript = False  # Convert model torchscript for inference
    args.opt = 'sgd'  # Optimizer (default: "sgd")
    args.opt_eps = None  # Optimizer Epsilon (default: None, use opt default)
    args.opt_betas = None  # Optimizer Betas (default: None, use opt default)
    args.momentum = 0.9  # Optimizer momentum (default: 0.9)
    args.weight_decay = 0.0001  # Weight decay (default: 0.0001)
    args.clip_grad = None  # Clip gradient norm (default: None, no clipping)
    args.clip_mode = 'norm'  # Gradient clipping mode. One of ("norm", "value", "agc")
    args.smoothing = 0.1  # Label smoothing (default: 0.1)
    args.drop = 0.0  # Dropout rate (default: 0.0)
    args.drop_path = None  # Drop path rate (default: None)
    args.drop_block = None  # Drop block rate (default: None)
    return args

if __name__ == '__main__':
    main()