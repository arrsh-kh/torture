#pragma once
#include "layer.h"
#include "tensor.h"

void dense_forward_layer(struct Layer* layer, Tensor* input, Tensor* output);
void dense_backward_layer(struct Layer* layer, Tensor* d_out, Tensor* d_input);
