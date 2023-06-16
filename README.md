#  Im-Promptu: In-Context Composition from Image Prompts

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)
![CUDA](https://img.shields.io/badge/cuda-v11.1.1-76b900)
![License](https://img.shields.io/badge/license-Clear%20BSD-green)



This is the official repository for the paper [Im-Promptu: In-Context Composition from Image Prompts](https://arxiv.org/abs/2305.17262). Website for a quick visual summary of this work can be found [here](https://jha-lab.github.io/impromptu/).
## Table of Contents
- [Im-Promptu: In-Context Composition from Image Prompts]

- [Table of Contents]
  - [Environment setup](#environment-setup)
  - [Datasets](#datasets)
  - [Pixel Transformation](#pixel-transformation)
  - [Monolithic Model](#monolithic-model)
    - [Training](#training)
  - [Patch Network](#patch-network)
    - [Training](#training-1)
  - [Object Centric Learner (OCL)](#object-centric-learner-ocl)
    - [Training](#training-2)
  - [Sequential Prompter](#sequential-prompter)
    - [Training](#training-3)
  - [Cite this work](#cite-this-work)
  - [License](#license)
## Environment setup

The following shell script creates an anaconda environment called "impromptu" and installs all the required packages. 
```shell
source env_setup.sh
```

## Datasets

The datasets can be downloaded from the [following]() link. The datasets should be placed in the `./datasets` directory. Detailed information about the benchmarks can be found in the `./benchmarks/README.md` file.

## Pixel Transformation

Solving analogies by simple transformation over the pixel space 
> $\hat{D} = C+ (B-A)$

A command instance to run the pixel transformation model
```shell
python3 learners/pixel.py --dataset shapes3d --batch_size 64 --data_path ./datasets/shapes3d/train.h5 --logs_dir ./logs_dir/ --phase val
```

The various arguments that can be passed to the script are:
```shell
--dataset = name of the dataset (options: shapes3d, clevr, bitmoji)

--batch_size = batch size for training

--data_path = path to the training data

--logs_dir = path to the directory where the logs will be stored

--phase = split of the dataset to evaluate on
```

## Monolithic Model

Monolithic vector representation to solve visual analogies. Architecture laid out in `./learners/monolithic.py`



### Training

Training instance of a monolithic learner is given below:

```
cd train_scripts

python train_monolithic.py --epochs 100 --dataset shapes3d --data_path ../datasets/shapes3d/train.h5 --image_size 64 --seed 0 --d_model 192 --logs_dir ../logs_dir/

```
Hyperparameters can be tweaked as follows
```
--epochs = Training epochs

--dataset = Name of the dataset to spawn Dataset from ./utils/create_dataset.py

--d_model = Latent vector dimension

--image_size = Input image size

--lr_main = Peak learning rate

--lr_warmup_steps = Learning rate warmup steps for linear schedule

--data_path = Path to the dataset

--log_path = path to the directory where the logs will be stored

```



## Patch Network

Patch abstractions to solve visual analogies. Architecture laid out in `./learners/patch_network.py`

### Training

```
cd train_scripts

python3 train_patch_network.py --batch_size 16 --dataset shapes3d --img_channels 3 --epochs 150 --data_path ./datasets/shapes3d/train.h5 --vocab_size 512 --image_size 64 --num_enc_heads 4 --num_enc_blocks 4 --num_dec_blocks 4 --num_heads 4 --seed 3
```
Additional hyperparameters are as follows
```shell
--vocab_size = Size of dVAE vocabulary

--num_dec_block = Number of decoder blocks

--num_enc_block = Number of context encoder blocks

--num_heads = Number of attention heads in the decoder

--num_enc_heads = Number of attention heads in the context encoder
```


## Object Centric Learner (OCL)
Solving analogies by learning object-centric representations. Architecture laid out in `./learners/object_centric_learner.py`

### Training

```
cd train_scripts/

python train.py  --img_channels 3 --dataset shapes3d --batch_size 32 --epochs 150 --data_path ./datasets/shapes3d/train.h5 --vocab_size 512 --image_size 64 --num_iterations 3 --num_slots 3 --num_enc_heads 4 --num_enc_blocks 4 --num_dec_heads 4 --num_heads 4 --slate_encoder_path ./logs_dir_pretrain/SLATE/best_encoder.pt --lr_warmup_steps 15000 --seed 0 --log_path ./logs_dir/
```

```
--num_slots = Number of object slots per image

--num_iterations = Number of iterations for slot attention

--slate_encoder_path = Path to the pre-trained slate encoder
```

## Sequential Prompter
Architecture laid out in `./learners/sequential_prompter.py`

### Training

```
cd train_scripts/

python train_prompt_.py  --img_channels 3 --epochs 150 --data_path ./datasets/shapes3d/train.h5 --vocab_size 512 --image_size 64 --num_iterations 3 --num_slots 3 --slate_encoder_path ./logs_dir_pretrain/shapes3d_SLATE/best_encoder.pt --seed 0
```


## Cite this work

Cite our work using the following bitex entry:
```bibtex
@misc{dedhia2023impromptu,
      title={Im-Promptu: In-Context Composition from Image Prompts}, 
      author={Bhishma Dedhia and Michael Chang and Jake C. Snell and Thomas L. Griffiths and Niraj K. Jha},
      year={2023},
      eprint={2305.17262},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

The Clear BSD License
Copyright (c) 2023, Bhishma Dedhia and Jha Lab.
All rights reserved.

See License file for more details.