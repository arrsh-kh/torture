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

void tensor_free(Tensor t) {
    free(t.data);
}
