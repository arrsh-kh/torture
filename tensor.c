#include<stdio.h>
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

int main() {
    Tensor t = tensor_alloc(1, 3, 4, 4);

    int i = IDX4(0, 1, 2, 3, t.c, t.h, t.w);
    t.data[i] = 42.0f;

    printf("Value: %f\n", t.data[i]); // should print 42.0

    tensor_free(t);
    return 0;
}
