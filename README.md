# MNIST Cls Cpp

A demo of using libtorch to train models on MNIST dataset for classification.

## Build

``` bash
cd MNIST-cls-cpp

cmake -S . -B build

cmake --build build
```

## Training and testing

``` bash
# train
./build/train -p path/to/mnist

# test
./build/test -p path/to/mnist
```
