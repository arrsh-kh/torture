#include <stdlib.h>
#include "tensor.h"

Tensor tensor_alloc(int n, int c, int h, int w) {
    Tensor t;
    t.n = n;
    t.c = c;
    t.h = h;
    t.w = w;
    int size = n * c * h * w;
    t.data = (float*)calloc(size, sizeof(float)); // zero-init 
    return t;
}

int tensor_size(Tensor* t) {
    return t->n * t->c * t->h * t->w;
}
void tensor_free(Tensor t) {
    free(t.data);
}
Tensor* tensor_alloc_ptr(int n, int c, int h, int w) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    *t = tensor_alloc(n, c, h, w);
    return t;
}

Tensor tensor_alloc_like(const Tensor* t) {
    return tensor_alloc(t->n, t->c, t->h, t->w);
}
