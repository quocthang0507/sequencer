import argparse
import logging
import os
from contextlib import suppress
from datetime import datetime

import torch.nn as nn
import yaml
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import *
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint
from timm.models.layers import convert_splitbn_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import utils.timm.checkpoint_saver as cs
import utils.timm.summary as sm
from train import *
from utils.timm.dataset_factory import create_dataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

try:
    import clearml

    has_clearml = True
except ImportError:
    has_clearml = False

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

must_download = True
device = "cuda:0"  # cuda:0 | cpu
_logger = logging.getLogger('train')
has_native_amp = False


def _parse_args():
    # Directly assign the values from the statement
    args = argparse.Namespace()
    if must_download:
        args.data_dir = "sequencer/datasets"  # Directory to store the dataset
        args.dataset = "torch/flowers"  # Dataset name
        args.dataset_download = True  # Flag to download the dataset
    else:
        args.data_dir = "sequencer/datasets/102flowers"  # Directory to store the dataset
        args.dataset = ""  # Dataset name
        args.dataset_download = False  # Flag to download the dataset
    args.model = "sequencer2d_s"  # Model name
    args.batch_size = 16  # Batch size for training
    args.workers = 2  # Number of data loading workers
    args.opt = "adamw"  # Optimizer type
    args.epochs = 300  # Number of epochs to train
    args.sched = "cosine"  # Learning rate scheduler type
    args.native_amp = True  # Use native automatic mixed precision
    args.img_size = 128  # Image size
    args.drop_path = 0.1  # Drop path rate
    args.lr = 2e-3  # Learning rate
    args.weight_decay = 0.05  # Weight decay
    args.remode = "pixel"  # Random erase mode
    args.reprob = 0.25  # Random erase probability
    args.aa = "rand-m9-mstd0.5-inc1"  # AutoAugment policy
    args.smoothing = 0.1  # Label smoothing
    args.mixup = 0.8  # Mixup alpha
    args.cutmix = 1.0  # Cutmix alpha
    args.warmup_lr = 1e-6  # Warmup learning rate
    args.warmup_epochs = 20  # Number of warmup epochs

    # Additional default values
    args.validation_batch_size = None  # Validation batch size (default: None)
    args.recount = 1  # Random erase count (default: 1)
    args.pretrained = False  # Use pretrained model (default: False)
    args.local_rank = 0  # Local rank for distributed training (default: 0)
    args.log_wandb = False  # Log metrics to wandb (default: False)
    args.log_clearml = False  # Log metrics to clearml (default: False)
    args.no_prefetcher = False  # Disable fast prefetcher (default: False)
    args.distributed = False  # Use distributed training (default: False)
    args.device = device  # Device to use for training (default: cuda:0)
    args.world_size = 1  # World size for distributed training (default: 1)
    args.rank = 0  # Global rank for distributed training (default: 0)
    args.prefetcher = not args.no_prefetcher  # Use fast prefetcher (default: True)
    args.no_resume_opt = False  # Prevent resume of optimizer state (default: False)
    args.resume = ''  # Path to resume checkpoint (default: '')
    args.initial_checkpoint = ''  # Path to initial checkpoint (default: '')
    args.num_classes = None  # Number of label classes (default: None)
    args.gp = None  # Global pool type (default: None)
    args.channels_last = False  # Use channels_last memory layout (default: False)
    args.sync_bn = False  # Enable synchronized BatchNorm (default: False)
    args.split_bn = False  # Enable separate BN layers per augmentation split (default: False)
    args.torchscript = False  # Convert model to torchscript (default: False)
    args.fuser = ''  # Select jit fuser (default: '')
    args.output = ''  # Path to output folder (default: '')
    args.experiment = ''  # Name of train experiment (default: '')
    args.eval_metric = 'top1'  # Best metric (default: 'top1')
    args.tta = 0  # Test/inference time augmentation factor (default: 0)
    args.use_multi_epochs_loader = False  # Use multi-epochs-loader (default: False)
    args.seed = 42  # Random seed (default: 42)
    args.worker_seeding = 'all'  # Worker seed mode (default: 'all')
    args.log_interval = 50  # Log interval (default: 50)
    args.recovery_interval = 0  # Recovery checkpoint interval (default: 0)
    args.checkpoint_hist = 10  # Number of checkpoints to keep (default: 10)
    args.save_images = False  # Save images of input batches (default: False)
    args.amp = False  # Use automatic mixed precision (default: False)
    args.apex_amp = False  # Use NVIDIA Apex AMP (default: False)
    args.no_ddp_bb = False  # Disable broadcast buffers for native DDP (default: False)
    args.pin_mem = False  # Pin CPU memory in DataLoader (default: False)
    args.train_split = 'train'  # Dataset train split (default: 'train')
    args.val_split = 'validation'  # Dataset validation split (default: 'validation')
    args.class_map = ''  # Path to class to idx mapping file (default: '')
    args.opt_eps = None  # Optimizer epsilon (default: None)
    args.opt_betas = None  # Optimizer betas (default: None)
    args.momentum = 0.9  # Optimizer momentum (default: 0.9)
    args.clip_grad = None  # Clip gradient norm (default: None)
    args.clip_mode = 'norm'  # Gradient clipping mode (default: 'norm')
    args.lr_noise = None  # Learning rate noise on/off epoch percentages (default: None)
    args.lr_noise_pct = 0.67  # Learning rate noise limit percent (default: 0.67)
    args.lr_noise_std = 1.0  # Learning rate noise std-dev (default: 1.0)
    args.lr_cycle_mul = 1.0  # Learning rate cycle len multiplier (default: 1.0)
    args.lr_cycle_decay = 0.5  # Amount to decay each learning rate cycle (default: 0.5)
    args.lr_cycle_limit = 1  # Learning rate cycle limit (default: 1)
    args.lr_k_decay = 1.0  # Learning rate k-decay for cosine/poly (default: 1.0)
    args.min_lr = 1e-6  # Lower lr bound for cyclic schedulers (default: 1e-6)
    args.epoch_repeats = 0.0  # Epoch repeat multiplier (default: 0.0)
    args.start_epoch = None  # Manual epoch number (default: None)
    args.decay_epochs = 100  # Epoch interval to decay LR (default: 100)
    args.cooldown_epochs = 10  # Epochs to cooldown LR at min_lr (default: 10)
    args.patience_epochs = 10  # Patience epochs for Plateau LR scheduler (default: 10)
    args.decay_rate = 0.1  # LR decay rate (default: 0.1)
    args.no_aug = False  # Disable all training augmentation (default: False)
    args.scale = [0.08, 1.0]  # Random resize scale (default: [0.08, 1.0])
    args.ratio = [3. / 4., 4. / 3.]  # Random resize aspect ratio (default: [3. / 4., 4. / 3.])
    args.hflip = 0.5  # Horizontal flip training aug probability (default: 0.5)
    args.vflip = 0.0  # Vertical flip training aug probability (default: 0.0)
    args.color_jitter = 0.4  # Color jitter factor (default: 0.4)
    args.aug_repeats = 0  # Number of augmentation repetitions (default: 0)
    args.aug_splits = 0  # Number of augmentation splits (default: 0)
    args.jsd_loss = False  # Enable Jensen-Shannon Divergence + CE loss (default: False)
    args.bce_loss = False  # Enable BCE loss w/ Mixup/CutMix use (default: False)
    args.bce_target_thesh = None  # Threshold for binarizing softened BCE targets (default: None)
    args.resplit = False  # Do not random erase first augmentation split (default: False)
    args.cutmix_minmax = None  # Cutmix min/max ratio (default: None)
    args.mixup_prob = 1.0  # Probability of performing mixup or cutmix (default: 1.0)
    args.mixup_switch_prob = 0.5  # Probability of switching to cutmix (default: 0.5)
    args.mixup_mode = 'batch'  # How to apply mixup/cutmix params (default: 'batch')
    args.mixup_off_epoch = 0  # Turn off mixup after this epoch (default: 0)
    args.train_interpolation = 'random'  # Training interpolation (default: 'random')
    args.drop = 0.0  # Dropout rate (default: 0.0)
    args.drop_connect = None  # Drop connect rate (default: None)
    args.drop_block = None  # Drop block rate (default: None)
    args.bn_momentum = None  # BatchNorm momentum override (default: None)
    args.bn_eps = None  # BatchNorm epsilon override (default: None)
    args.dist_bn = 'reduce'  # Distribute BatchNorm stats between nodes (default: 'reduce')
    args.model_ema = False  # Enable tracking moving average of model weights (default: False)
    args.model_ema_force_cpu = False  # Force ema to be tracked on CPU (default: False)
    args.model_ema_decay = 0.9998  # Decay factor for model weights moving average (default: 0.9998)
    args.task_name = ''  # Name of train task (default: '')
    args.output_uri = ''  # URI to save weights of model (default: '')
    args.log_s3 = False  # Log weights to s3 (default: False)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    main()
