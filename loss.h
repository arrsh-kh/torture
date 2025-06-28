#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

float loss_mse(Tensor* prediction, Tensor* target);
void loss_mse_deriv(Tensor* prediction, Tensor* target, Tensor* grad_out);

#endif
