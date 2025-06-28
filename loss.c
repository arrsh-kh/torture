#include "tensor.h"
float loss_mse(Tensor* prediction, Tensor* target) {
    float sum = 0.0f;
    int n = prediction->w;  // assuming 1×1×1×n
    for (int i = 0; i < n; i++) {
        float diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / n;
}

void loss_mse_deriv(Tensor* prediction, Tensor* target, Tensor* grad_out) {
    int n = prediction->w;
    for (int i = 0; i < n; i++) {
        grad_out->data[i] = 2.0f * (prediction->data[i] - target->data[i]) / n;
    }
}
