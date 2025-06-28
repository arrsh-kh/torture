#include <math.h>
#include "activation.h"

void tanh_forward(Layer* layer, Tensor* input, Tensor* output) {
    int size = tensor_size(input);
    for (int i = 0; i < size; i++)
        output->data[i] = tanhf(input->data[i]);
    layer->input_cache = input;
}

void tanh_backward(Layer* layer, Tensor* d_out, Tensor* d_input) {
    int size = tensor_size(d_out);
    for (int i = 0; i < size; i++) {
        float x = layer->input_cache->data[i];
        d_input->data[i] = (1.0f - x * x) * d_out->data[i];
    }
}
