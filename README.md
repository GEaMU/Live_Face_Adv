# **Adversarial Example Generation for Live Face Recognition Based on 3D Gaussian Modeling and Diffusion Models**

![](README.assets/result3D.svg)

## Install

- Build environment

```bash
conda env create -f env.yaml
```

- Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d).


```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.htm
```

- Install [kaolin](https://github.com/NVIDIAGameWorks/kaolin).

```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html
```

- Install diff-gaussian-rasterization and simple_knn from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). Note, for rendering 32-channel images, please modify "NUM_CHANNELS 3" to "NUM_CHANNELS 32" in "diff-gaussian-rasterization/cuda_rasterizer/config.h".

- Download ["tets_data.npz"](https://drive.google.com/file/d/1SMkp8v8bDyYxEdyq25jWnAX1zeQuAkNq/view?usp=drive_link) and put it into "assets/".

- Download [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/), and progress refer to [3DMM](https://github.com/YuelangX/Multiview-3DMM-Fitting).

## Training 

```
python train_stage1.py --config config/train_stage1.yaml
python train_stage2.py --config config/train_stage2.yaml
```

## Generate dynamic gaussian avatars

After selecting the target facial expression and action, process them according to the **Prepare for Gaussian Head Avatar** and then reproduce the target expression.

```
python generate_exp.py --config config/generate_exp.yaml
```

## Generate adversarial example

```
bash generate_adv.sh
```