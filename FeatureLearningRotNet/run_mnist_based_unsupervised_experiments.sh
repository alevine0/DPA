#!/bin/bash

# Train a RotNet (with a NIN architecture of 4 conv. blocks) on training images of MNIST.
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp=MNIST_RotNet_NIN4blocks

