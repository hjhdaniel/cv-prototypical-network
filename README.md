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

## Required Datasets (Download Manually)

1) **Mini-imagenet**: [Download link](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)

## Dataset Directories

- dataset
  - imagenet
    - images **(PUT IMAGENET IMAGES HERE)**
    - materials
  - cub200
  - omniglot

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
10) feature_proto.py: main code to run model with extraction of features

## Performances

To reproduce the results below:

| Model | 5-shot (5-way Acc.) | 1 -shot (5-way Acc.) | 1-shot (20-way Acc.)|
| --- | --- | --- | --- |
| Original | 99.7% | 98.8% | 94.7% |
| Feature Extraction | 99.5%* | 98.3%°| 94.6%°° |

\* achieved running `python feature_proto.py --cuda` (default)

° achieved running `python feature_proto.py --cuda -nsTr 1 -nsVa 1`

°° achieved running `python feature_proto.py --cuda -nsTr 1 -nsVa 1 -cVa 20`

## Improvements and associated branches

| Improvements                             | Branch Name                             |
| ---------------------------------------- | --------------------------------------- |
| Embedding Architecture                   | embedding-architecture                  |
| Exploration of New Distance Functions    | new-distance-functions                  |
| Soft clustering with Gaussian prototypes | gaussian-prototypes<br />gaussian-paper |
| Feature Extraction for Embeddings        | feature-extraction-embeddings           |
| New Datasets                             | new-datasets                            |
