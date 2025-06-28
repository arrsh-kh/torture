#pragma once
#include "tensor.h"

void conv2d_forward(
    Tensor* input,
    Tensor* weights,
    Tensor* bias,
    Tensor* output,
    int stride,
    int padding
); 
