#include "dense.h"
#include <stdlib.h>

void dense_forward_layer(Layer* layer, Tensor* input, Tensor* output) {
    int in_size = input->w;
    int out_size = layer->weights->n;

    output->n = 1;
    output->c = 1;
    output->h = 1;
    output->w = out_size;
    output->data = malloc(sizeof(float) * out_size);

    for (int o = 0; o < out_size; o++) {
        float sum = layer->bias->data[o];
        for (int i = 0; i < in_size; i++) {
            sum += input->data[i] * layer->weights->data[o * in_size + i];
        }
        output->data[o] = sum;
    }

    layer->input_cache = input;
}

void dense_backward_layer(Layer* layer, Tensor* d_out, Tensor* d_input) {
    int in_size = layer->weights->c * layer->weights->h * layer->weights->w;
    int out_size = layer->weights->n;

    d_input->n = 1;
    d_input->c = 1;
    d_input->h = 1;
    d_input->w = in_size;
    d_input->data = malloc(sizeof(float) * in_size);

    for (int i = 0; i < in_size; i++) {
        float grad = 0.0f;
        for (int o = 0; o < out_size; o++) {
            grad += d_out->data[o] * layer->weights->data[o * in_size + i];
            float dw = layer->input_cache->data[i] * d_out->data[o];
            layer->weights->data[o * in_size + i] -= 0.01f * dw;
        }
        d_input->data[i] = grad;
    }

    for (int o = 0; o < out_size; o++) {
        layer->bias->data[o] -= 0.01f * d_out->data[o];
    }
}
