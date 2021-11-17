# DIVeR: Real-time CUDA Application
This repo contains the code for the real-time application of DIVeR implemented in Python+CUDA.

## Setup

### Pre-trained model

We have two versions of pre-trained models trained on nerf_synthetic dataset. 

- Our [256 model](https://drive.google.com/file/d/1dEpMamHreZVtKV9BZA9uGFUJKQbBdJFq/view?usp=sharing) (256x256x256 voxel grid).
- Our [128 model](https://drive.google.com/file/d/11p0XdSNQrp_9HDbvQZaS7s9LZDi_v3QH/view?usp=sharing) (128x128x128 voxel grid).

The 128 model runs much faster with smaller storage cost, but the 256 model has better rendering quality.

## Usage

To launch the real-time application, run:

```shell
python run.py --weight_path=PATH_TO_WEIGHT_FILE \
			  --voxel_num=VOXEL_GRID_SIZE \
			  --device=GPU_DEVICE \
```

 ## Resources

- Project page
- Paper
- [Training code](https://github.com/lwwu2/diver)

## Citation

