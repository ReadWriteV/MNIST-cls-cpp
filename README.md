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
./build/train -p path/to/mnist

# test
./build/test -p path/to/mnist
```
