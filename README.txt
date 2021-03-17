Code for the paper "Deep Partition Aggregation: Provable Defense against General Poisoning Attacks" by Alexander Levine and Soheil Feizi.

This directory contains the code necessary to train and test models which are provably robust to poisoning attacks. We use python3.7, PyTorch, and torchvision, and assume CUDA capabilities.


++++++Training++++++

DPA: 
	To train a DPA ensemble (for defense against general poisoning attacks) on {MNIST, CIFAR, GTSRB}, use the file 'train_{mnist,cifar,gtsrb}_nin_baseline.py' (nin_baseline stands for 'Network-In-Network', the classifier architecture that we use).  In order to train large ensembles, the base classifiers can be trained in parallel; however, we must first create training set partitions, using 'partition_data_norm_hash.py'. Partitions have already been created for the experiments in the paper ('partitions_hash_mean_cifar_1000.pth', etc.). However, for completeness, we document a full example here:

	python3 partition_data_norm_hash.py --dataset cifar --partitions 1000

	Then, possibly in parallel:

	python3 train_cifar_nin_baseline.py --num_partitions 1000 --num_partition_range 500 --start_partition 0
	python3 train_cifar_nin_baseline.py --num_partitions 1000 --num_partition_range 500 --start_partition 500

	Trained ensembles will be directories in the 'checkpoints' directory.

SS-DPA (RotNet / MNIST):
	To train an SS-DPA ensemble (for defense against label-flip attacks) on MNIST, we need a pre-trained feature extractor. We include a slightly modified (deterministic) version of the RotNet (Gidaris et al, ICLR 2018) feature extractor training system, and provide pre-trained feature extractor models for MNIST. However, if you want to do it yourself:
			First, for determinism purposes, we need to create an ordered version of the dataset. In the 'FeatureLearningRotNet' directory, run:

			python3 order_dataset_for_unsupervised.py --dataset mnist

			Then train the feature extractor:

			./run_mnist_based_unsupervised_experiments.sh

	Now, we again need to create partitions before training. These are provided, but you can run:

	python3 partition_data_unsupervised.py --dataset mnist --partitions 1200

	This will create a file 'partitions_unsupervised_mnist_1200.pth.'  Then, to train, possibly in parallel:

	python3 train_mnist_rotnet.py --num_partitions 1200 --num_partition_range 600 --start_partition 0
	python3 train_mnist_rotnet.py --num_partitions 1200 --num_partition_range 600 --start_partition 600

	Note that you can use the hashing based partitions ('partitions_hash_mean_mnist_1200.pth', etc.) instead, to reproduce Appendix E, using the '--hash_partitions' option.

	Trained ensembles will be directories in the 'checkpoints' directory. 


++++++Evaluation+++++

To evaluate models, use, for example:

python3 evaluate_mnist_rotnet.py --models mnist_rotnet_partitions_1200
python3 certify.py --evaluations mnist_rotnet_partitions_1200.pth


Where 'mnist_rotnet_partitions_1200' is a directory (representing a trained ensemble) in the 'checkpoints' directory. The first line will create a file 'mnist_rotnet_partitions_1200.pth' in the 'evaluations' directory, which contains the results of running every partition model on every test sample.  The second line will read this file, and output in the 'radii' directory another file called 'mnist_rotnet_partitions_1200.pth'. This is a PyTorch save file consisting of a 1-dimensional tensor, where every element is the certified robustness (symmetric difference or label flips, depending upon the model) of a test sample, for the entire test set. Misclassified samples are marked with robustness of -1. This will also print the average base classifier accuracy, smoothed classifier accuracy, and the median certified robustness as seen in Table 1.

++++++ SS-DPA (SimCLR / CIFAR and GTSRB) -- Training and Evaluation ++++++++

	For SS_DPA with SimCLR embeddings, see the SupContrast directory (Forked from https://github.com/HobbitLong/SupContrast, which provides SimCLR as well as SupContrast algorithms). Specifically, see (in order):

	train_embedding_cifar.sh (Trains embedding; Note that these are provided for CIFAR-10 and GTSRB.)

	train_cifar_ensemble_example.sh (Trains ensemble of size 250; in ./SupContrast/checkpoints/)

	evaluate_cifar_simclr.py

	certify.py

++++++Additional Features++++

If you use the '--zero_seed' option for any of the 'train_' or 'evaluate_' scripts, they will use zero as the random seed, instead of the partition number (Appendix F). The Binary MNIST experiment (Appendix B) can be run as:

python3 cluster_binary_mnist.py


