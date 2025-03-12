Install torch:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install VC++ Build Tools: [https://github.com/bycloudai/InstallVSBuildToolsWindows](https://github.com/bycloudai/InstallVSBuildToolsWindows)

Install [cuda_12.4.0_551.61_windows](https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe)

Download [cudnn-windows-x86_64-9.6.0.74_cuda12-archive](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.6.0.74_cuda12-archive.zip), and unpack it, copy `<CUDNN_zip_files_path>\cuda\bin\cudnn64_9.dll` to `<YOUR_DRIVE>\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`.

Check CUDA driver:

```
>>> import pytorch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
```

Install required packages:
```
pip install -r requirements.txt
```

Update ClearML:

```
pip install --upgrade clearml
```

Download the dataset and train the model:

```
python train.py E:\GitHub\sequencer\datasets --dataset torch/flowers --dataset-download --model sequencer2d_s -b 8 -j 8 --opt adamw --epochs 300 --sched cosine --native-amp --img-size 128 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```

Train the model with the existed dataset:

```
python train.py E:\GitHub\sequencer\datasets\102flowers --model sequencer2d_s -b 16 -j 4 --opt adamw --epochs 300 --sched cosine --native-amp --img-size 128 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```
