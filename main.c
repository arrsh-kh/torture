#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "conv2d.h"
#include "dense.h"
#include "flatten.h"
#include "activation.h"
#include "loss.h"
#include "layer.h"

#define NUM_LAYERS 4

int main() {
    float inputs[3][16] = {
        { 0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0 },
        { 0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0 },
        { 0,0,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0 }
    };

    float targets[3][2] = {
        { 1, 0 },
        { 0, 1 },
        { 0, 1 }
    };

    Layer model[NUM_LAYERS];

    model[0].type = LAYER_CONV2D;
    model[0].stride = 1;
    model[0].padding = 0;
    model[0].weights = tensor_alloc_ptr(1, 1, 3, 3);
    model[0].bias = tensor_alloc_ptr(1, 1, 1, 1);
    model[0].forward = conv2d_forward_layer;
    model[0].backward = conv2d_backward_layer;

    model[1].type = LAYER_TANH;
    model[1].forward = tanh_forward;
    model[1].backward = tanh_backward;

    model[2].type = LAYER_FLATTEN;
    model[2].forward = flatten_forward;
    model[2].backward = flatten_backward;

    model[3].type = LAYER_DENSE;
    model[3].weights = tensor_alloc_ptr(2, 4, 1, 1);
    model[3].bias = tensor_alloc_ptr(1, 1, 1, 2);
    model[3].forward = dense_forward_layer;
    model[3].backward = dense_backward_layer;

    for (int i = 0; i < 9; i++) model[0].weights->data[i] = 1;
    model[0].bias->data[0] = 0.0f;
    for (int i = 0; i < 8; i++) model[3].weights->data[i] = 0.1f * (i + 1);
    model[3].bias->data[0] = 0.5f;
    model[3].bias->data[1] = -0.5f;

    float lr = 0.01f;
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss_total = 0.0f;
        for (int sample = 0; sample < 3; sample++) {
            Tensor input = tensor_alloc(1, 1, 4, 4);
            for (int i = 0; i < 16; i++) input.data[i] = inputs[sample][i];

            Tensor target = tensor_alloc(1, 1, 1, 2);
            for (int i = 0; i < 2; i++) target.data[i] = targets[sample][i];

            Tensor* current = &input;
            Tensor outputs[NUM_LAYERS];

            for (int l = 0; l < NUM_LAYERS; l++) {
                Tensor* out = tensor_alloc_ptr(1, 1, 1, 1);
                model[l].forward(&model[l], current, out);
                outputs[l] = *out;
                current = out;
            }

            float loss = loss_mse(current, &target);
            loss_total += loss;

            Tensor grad = tensor_alloc(1, 1, 1, 2);
            loss_mse_deriv(current, &target, &grad);

            Tensor* grad_in = tensor_alloc_ptr(1, 1, 1, 2);
            for (int i = 0; i < 2; i++) grad_in->data[i] = grad.data[i];

            for (int l = NUM_LAYERS - 1; l >= 0; l--) {
                Tensor* grad_out = tensor_alloc_ptr_like(&outputs[l]);  // ✅ Now it's a pointer
                model[l].backward(&model[l], grad_in, grad_out);
                tensor_free(*grad_in);
                free(grad_in);
                grad_in = grad_out;
            }
tensor_free(*grad_in);
free(grad_in);

            tensor_free(input);
            tensor_free(target);
            tensor_free(grad);
            for (int l = 0; l < NUM_LAYERS; l++) tensor_free(outputs[l]);
        }

        if (epoch % 10 == 0 || epoch == epochs - 1)
            printf("[Epoch %3d] Avg Loss: %.6f\n", epoch, loss_total / 3.0f);
    }

    tensor_free(*model[0].weights);
    tensor_free(*model[0].bias);
    tensor_free(*model[3].weights);
    tensor_free(*model[3].bias);

    return 0;
}
