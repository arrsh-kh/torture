#pragma once

typedef struct {
    float* data;
    int n, c, h, w;
} Tensor;

Tensor* tensor_alloc_ptr(int n, int c, int h, int w);
Tensor tensor_alloc(int n, int c, int h, int w);
void tensor_free(Tensor t);
int tensor_size(Tensor* t);
Tensor tensor_alloc_like(const Tensor* t);
#define IDX4(n, c, h, w, C, H, W) (((n)*(C)*(H)*(W)) + (c)*(H)*(W) + (h)*(W) + (w)) //calculates tensor location
