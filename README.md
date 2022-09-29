# MNIST Cls Cpp

A demo of using libtorch to train and test model on MNIST dataset for classification.

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

## Benchmark

Epoch: 5, Batch size: 32, Learing rate: 0.01.

|            | Correct% | Train Time |
| ---------- | -------- | ---------- |
| Simple Net |   88.66  |   13723ms  |
|  LeNet-5   |   99.05  |   56465ms  |
