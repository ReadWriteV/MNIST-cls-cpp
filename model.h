#pragma once

#include <torch/torch.h>

#include "model.h"

/// @brief a simple classifier with 3 FC layer for mnist dataset
class MNIST_ClsImpl : public torch::nn::Module
{
public:
    /// @brief Constructor
    /// @param in_dim the dimension of input
    /// @param n_hidden_1 the dimension of hidden layer 1
    /// @param n_hidden_2 the dimension of hidden layer 2
    /// @param out_dim the dimension of output
    MNIST_ClsImpl(int in_dim, int n_hidden_1, int n_hidden_2, int out_dim);

    /// @brief forward function
    /// @param x input tensor
    /// @return output tensor
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
};

TORCH_MODULE(MNIST_Cls);
