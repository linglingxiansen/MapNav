# Mapnav: A novel memory representation via annotated semantic maps for vlm-based vision-and-language navigation (ACL 2025)
Repository for **Mapnav: A novel memory representation via annotated semantic maps for vlm-based vision-and-language navigation** (ACL 2025)

<h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-9BDFDF)](https://github.com/linglingxiansen/MapNav/blob/main/LICENSE) 
[![hf_checkpoint](https://img.shields.io/badge/🤗-Dataset-FBD49F.svg)](https://huggingface.co/datasets/llxs/MapNav)
[![arXiv](https://img.shields.io/badge/Arxiv-2508.14160-E69191.svg?logo=arXiv)](https://arxiv.org/abs/2502.13451) 



## Installation
The code has been tested only with Python 3.8 on Ubuntu 20.04.

1. Environments Setup
- Follow [L3MVN](https://raw.githubusercontent.com/ybgdgh/L3MVN/) to install Habitat-lab, Habitat-sim, rednet, torch and other independences.
- Install the [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT).

2. Dataset
- Download Matterport3d scene dataset to the data path.

3. Path
- Change the dataset path and habitat path in the config_utils.py

## Training
You can download the huggingface dataset to generate you QA pairs to train your own model using [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).
- [hf_checkpoint](https://huggingface.co/datasets/llxs/MapNav)
- [LLaVA-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

## Evaluation
CUDA_VISIBLE_DEVICES=1 python r2rnav_benchmark.py --split val1 --eval 1 --auto_gpu_config 0 -n 1 --num_local_steps 10 --print_images 1 --model_dir model_path --exp_name nohis_rgb --eval_episodes 1839 --collect 0 --stop_th 300

## ASM Generation
You can look for the generation and annotation pipeline in the r2rnav_benchmark.py and huatu3.py.
