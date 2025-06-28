#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"


void tanh_forward(Layer* layer, Tensor* input, Tensor* output);
void tanh_backward(Layer* layer, Tensor* d_out, Tensor* d_input);

#endif
