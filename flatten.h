#pragma once
#include "layer.h"
#include "tensor.h"

void flatten_forward(struct Layer* layer, Tensor* input, Tensor* output);
void flatten_backward(struct Layer* layer, Tensor* d_out, Tensor* d_input);
