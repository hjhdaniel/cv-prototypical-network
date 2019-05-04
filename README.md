# Computer Vision: Few Shot Learning With Prototypical Network

Reference Paper Title: Prototypical Networks for Few-shot Learning  
Reference Paper URL: https://arxiv.org/pdf/1703.05175v2.pdf  
Reference Paper Authors: Jake Snell, Kevin Swersky, Richard S. Zemel  

Edits of prototypical network code taken from: Orobix  
Code URL: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch  
Code Author: Daniele E. Ciriello  

## Required Packages

1) python3
2) numpy
3) pytorch
4) pillow
5) argparse

## Instruction to setup datasets

Setup **Mini-imagenet** by running the following or download manually from [Download link](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE):

```bash
cd scripts/ && bash download_imagenet.sh
```

Setup **CUB2011** by running the following:

```bash
cd scripts/ && python split_cub2011.py
```

Setup **cifar100** by download dataset manually from [CIFAR-100 python version](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and put it in dataset/CIFAR/, then run following:

```bash
cd scripts/ && python extract_cifar100.py
python split_cifar100.py
```



## Dataset Directories

- dataset
  - omniglot
  - imagenet
  - cub2011
  - cifar100

## Files

1) prototype_network.py: prototype network class module
2) batch_sampler.py: prototype batch sampler for episodic training within an epoch
3) dataloader.py: dataloader for network (add new/different datasets here)
4) loss_function.py: loss function for network
5) train.py: train module
6) test.py: test module
7) parser.py: parser to handle arguments in commandline
8) main.py: main code to run (edit this to change parameters for training/testing)
9) omniglot_dataset.py: for loading omniglot dataset