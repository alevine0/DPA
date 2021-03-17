#!/bin/bash

for PART in {0..249}
do
	python3 main_linear.py --batch_size 512 \
	   --learning_rate 1 \
	   --model resnet18 \
	   --dataset cifar10subset \
	   --num_workers 1 \
	   --num_partitions 250 \
	   --partition $PART \
	   --ckpt ./save/SupCon/cifar10_models/SimCLR_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_1_cosine_warm/last.pth
done