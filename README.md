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

## Improvements and associated branches

| Improvements                             | Branch Name                             |
| ---------------------------------------- | --------------------------------------- |
| Embedding Architecture                   | embedding-architecture                  |
| Exploration of New Distance Functions    | new-distance-functions                  |
| Soft clustering with Gaussian prototypes | gaussian-prototypes<br />gaussian-paper |
| Feature Extraction for Embeddings        | feature-extraction-embeddings           |
| New Datasets                             | new-datasets                            |

*note: gaussian-prototypes is our original implementation, gaussian-paper is an implementation based on this paper >> https://arxiv.org/abs/1708.02735. More details of each improvements can be found in each branch's README and in the report.*

## Changes

* implemented 2 different embedding architecture
  * Pretrained Resnet18
  * 5 convolutional blocks and 2 fully-connected layers
* changes can be found in `prototype_network.py`

## Performances

| Model | Embedding space size | Episodes | Training classes per episode | Test accuracy |
| --- | --- | --- | --- | --- |
| Original model (4 conv) | 1600 | 100 | 8 | 67.0% |
| Deeper Model (5 conv + 2 fc) | 500 | 100 | 8 | 69.4% |
| Resnet18 | 1500 | 100 | 8 | 70.1% |
