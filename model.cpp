#include "model.h"

MNIST_ClsImpl::MNIST_ClsImpl(int in_dim, int n_hidden_1, int n_hidden_2, int out_dim)
{
    layer1 = register_module("layer1", torch::nn::Linear(torch::nn::LinearOptions(in_dim, n_hidden_1)));
    layer2 = register_module("layer2", torch::nn::Linear(torch::nn::LinearOptions(n_hidden_1, n_hidden_2)));
    layer3 = register_module("layer3", torch::nn::Linear(torch::nn::LinearOptions(n_hidden_2, out_dim)));
}

torch::Tensor MNIST_ClsImpl::forward(torch::Tensor x)
{
    x = layer1(x);
    x = layer2(x);
    x = layer3(x);
    return x;
}