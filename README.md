# [NeurIPS 2023] Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks

* Under refactoring.

## Introduction
This Git repository provides the ***Softmax output approximation function***, which is an open-source code of [NeurIPS 2023](https://nips.cc/Conference/2023) paper titled "***Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks***".

This repository provides the proposed ***softmax output approximation function*** in Python/Pytorch, and demonstrates an example of a machine translation task using the Transformer and Multi30k dataset, as experimented in the paper.

## Software Install and Code Cloning
The Approximation function is implemented based on Python and Pytorch with a GPU. 

**Step 1.** Install [Python (>= 3.8)](https://www.python.org/downloads/).

**Step 2.** Install [Pytorch >= 1.12.1)](https://www.pytorch.org/).

**Step 3.** Clone this Softmax output approximation repository.
```sh
> git clone https://github.com/eai-lab/SoftmaxOutputApproximation.git
Cloning into 'SoftmaxOutputApproximation'...
remote: Enumerating objects: 7, done.
remote: Counting objects: 100% (7/7), done.
remote: Compressing objects: 100% (5/5), done.
remote: Total 7 (delta 0), reused 0 (delta 0), pack-reused 0
Receiving objects: 100% (7/7), 5.71 KiB | 5.71 MiB/s, done.
```

## How to use the Softmax output approximation function
**Step 1.** Import our fucntion along with the user-specific hyperparameter ***m***
```
from approximation_method import *
```
**Step 2.** Decide how many elements to select, considering the total length of the sentence in transformer_train.py.

**Step 3.** Replace the softmax function used in the attention mechanism with our softmax output approximation function.
```
attention = softmax_approximation.apply(energy, mask)
```

## Citation (BibTeX)
**[ Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks]**
```
@inproceedings{
lee2023softmax,
title={Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks},
author={Changhyeon Lee and Seulki Lee},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=oScaeIibRx}
}
```
