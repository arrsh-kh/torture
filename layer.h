#pragma once
#include "tensor.h"

typedef enum {
    LAYER_CONV2D,
    LAYER_DENSE,
    LAYER_TANH,
    LAYER_FLATTEN
} LayerType;

typedef struct Layer {
    LayerType type;

    void (*forward)(struct Layer* layer, Tensor* input, Tensor* output);
    void (*backward)(struct Layer* layer, Tensor* d_out, Tensor* d_input);

    Tensor* weights;
    Tensor* bias;

    Tensor* input_cache;
    Tensor* output_cache;

    int stride;
    int padding;
} Layer;
