#!/usr/bin/env python3
""" ImageNet Validation Script
This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.
Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import torch.nn.parallel

from validate import *
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
    args.data = "datasets/imagenet"  # Path to dataset
    args.model = "sequencer2d_s"  # Model architecture
    args.batch_size = 16  # Mini-batch size
    args.input_size = None  # Input image dimensions (channels, height, width)
    args.amp = False  # Use AMP mixed precision
    args.dataset = ""  # Dataset type (default: ImageFolder/ImageTar if empty)
    args.split = "validation"  # Dataset split
    args.dataset_download = False  # Disable dataset download
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
    args.pretrained = True  # Use pre-trained model
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

def main():
    setup_default_logging()
    args = _parse_args()
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
