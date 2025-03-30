#!/usr/bin/env python3
""" ImageNet Validation Script
This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.
Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from advertorch.attacks import GradientSignAttack, LinfPGDAttack
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy, random_seed

import models
from utils.helpers import NormalizeByChannelMeanStd, WithNone, train_rnn
from validation import validate, write_results
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

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

def _parse_args():
    args = argparse.Namespace()
    args.data = "datasets"  # Path to dataset
    args.model = "sequencer2d_s"  # Model architecture
    args.batch_size = 16  # Mini-batch size
    args.input_size = [3, 224, 224]  # Input image dimensions (channels, height, width)
    args.amp = True  # Use AMP mixed precision
    args.dataset = "torch/imagenet"  # Dataset type (default: ImageFolder/ImageTar if empty)
    args.split = "validation"  # Dataset split
    args.dataset_download = True  # Disable dataset download
    args.workers = 2  # Number of data loading workers
    args.img_size = None  # Input image dimension (uses model default if empty)
    args.crop_pct = None  # Input image center crop percentage
    args.mean = None  # Override mean pixel value of dataset
    args.std = None  # Override standard deviation of dataset
    args.interpolation = ""  # Image resize interpolation type
    args.num_classes = None  # Number of classes in dataset
    args.class_map = ""  # Path to class-to-index mapping file
    args.gp = None  # Global pool type
    args.log_freq = 10  # Batch logging frequency
    args.checkpoint = ""  # Path to latest checkpoint
    args.pretrained = False  # Use pre-trained model
    args.num_gpu = 1  # Number of GPUs to use
    args.test_pool = False  # Enable test time pool
    args.no_prefetcher = False  # Enable fast prefetcher
    args.pin_mem = False  # Pin CPU memory in DataLoader
    args.channels_last = False  # Use channels_last memory layout
    args.apex_amp = False  # Use NVIDIA Apex AMP mixed precision
    args.native_amp = True  # Use Native Torch AMP mixed precision
    args.tf_preprocessing = False  # Use TensorFlow preprocessing pipeline
    args.use_ema = False  # Use EMA version of weights if present
    args.torchscript = False  # Convert model to TorchScript for inference
    args.legacy_jit = False  # Use legacy JIT mode for older PyTorch versions
    args.results_file = ""  # Output CSV file for validation results
    args.real_labels = ""  # Real labels JSON file for ImageNet evaluation
    args.valid_labels = ""  # Valid label indices file for partial label space
    args.attack_type = None  # Attack type (e.g., 'fgsm', 'pgd')
    args.adv_eps = 1.0  # Adversarial attack epsilon
    args.adv_steps = 5  # Adversarial attack steps
    args.adv_step_size = 0.5  # Adversarial attack step size
    args.seed = 42  # Random seed
    return args
    
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.attack_type:
        normalize = NormalizeByChannelMeanStd(
            mean=data_config['mean'],
            std=data_config['std']
        )
        model = nn.Sequential(normalize, model)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'] if args.attack_type is None else (0, 0, 0),
        std=data_config['std'] if args.attack_type is None else (1, 1, 1),
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with WithNone() if args.attack_type else torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()

        if args.attack_type == "fgsm":
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
            adversary = GradientSignAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps / 255.,
                clip_min=0., clip_max=1., targeted=False)
        elif args.attack_type == "pgd":
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
            adversary = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps / 255.,
                nb_iter=args.adv_steps, eps_iter=args.adv_step_size / 255., rand_init=True,
                clip_min=0., clip_max=1., targeted=False)
        random_seed(args.seed, 0)

        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.attack_type:
                train_rnn(model)
                input = adversary.perturb(input, target)
                model.eval()
                input = input.detach()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # compute output
            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*_dino'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)

if __name__ == '__main__':
    main()
