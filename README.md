# MNIST Cls Cpp

A demo of using libtorch to train and test models on MNIST dataset for classification.

## Build

``` bash
cd MNIST-cls-cpp

cmake -S . -B build

cmake --build build
```

## Train and test

``` bash
# train
./build/train -p path/to/mnist/dataset

# test
./build/test -p path/to/mnist/dataset -m path/to/saved/model
```

## Included Models

### Simple Net

A simple classifier with three Fully Connected Layer.

### LeNet 5

Implement of the classical model **LeNet5**, according to *LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.*

### AlexNet

Implement of the classical model **AlexNet**, according to *Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C] International Conference on Neural Information Processing Systems. Curran Associates Inc. 2012:1097-1105. with a little modified.*

## Benchmark

Epoch: 5, Batch size: 32, Learing rate: 0.01.

The *Train Time* is measured on the train set with CPU i5-9300H, and the *Correct%* is measured on the test set.

|            | Correct% | Train Time |
| ---------- | -------- | ---------- |
| Simple Net |   88.66  |   13723ms  |
|  LeNet-5   |   99.05  |   56465ms  |
|  AlexNet   |   90.42  |  too long  |
