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
    args.batch_size = 16  # Batch size for training, default = 256
    args.workers = 2  # Number of data loading workers, default = 4
    args.opt = "adamw"  # Optimizer type
    args.epochs = 300  # Number of epochs to train
    args.sched = "cosine"  # Learning rate scheduler type
    args.native_amp = True  # Use native automatic mixed precision
    args.img_size = 128  # Image size, default = 224
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


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    if device == "cpu":
        model.cpu()
    elif device == "cuda:0":
        model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        if args.task_name:
            exp_name = os.path.join(exp_name, args.task_name)
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = cs.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist,
            log_clearml=args.log_clearml, log_s3=args.log_s3
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                sm.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb,
                    log_clearml=args.log_clearml and has_clearml)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


if __name__ == '__main__':
    main()
