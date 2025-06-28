#include "flatten.h"
#include <stdlib.h>

void flatten_forward(Layer* layer, Tensor* input, Tensor* output) {
    int size = input->n * input->c * input->h * input->w;
    output->n = input->n;
    output->c = 1;
    output->h = 1;
    output->w = size;
    output->data = malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        output->data[i] = input->data[i];
    }
    layer->input_cache = input;
}

void flatten_backward(Layer* layer, Tensor* d_out, Tensor* d_input) {
    int size = d_out->n * d_out->w;
    d_input->n = layer->input_cache->n;
    d_input->c = layer->input_cache->c;
    d_input->h = layer->input_cache->h;
    d_input->w = layer->input_cache->w;
    d_input->data = malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        d_input->data[i] = d_out->data[i];
    }
}
