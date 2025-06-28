#pragma once
#include "tensor.h"
#include "layer.h"
void conv2d_forward_layer(Layer* layer, Tensor* input, Tensor* output);
void conv2d_backward_layer(Layer* layer, Tensor* d_out, Tensor* d_input);
