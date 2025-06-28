// gore ahead proceed with caution 
#include "conv2d.h"
#include "tensor.h"

void conv2d_forward(
    Tensor* input,
    Tensor* weights,
    Tensor* bias,
    Tensor* output,
    int stride,
    int padding
) {
    int batch_size = input->n;
    int in_c = input->c;
    int in_h = input->h;
    int in_w = input->w;

    int out_c = weights->n;
    int kernel = weights->h;

    int out_h = output->h;
    int out_w = output->w;

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {

                    float sum = bias->data[oc];

                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < kernel; kh++) {
                            for (int kw = 0; kw < kernel; kw++) {

                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = IDX4(b, ic, ih, iw, in_c, in_h, in_w);
                                    int w_idx = IDX4(oc, ic, kh, kw, in_c, kernel, kernel);
                                    sum += input->data[in_idx] * weights->data[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = IDX4(b, oc, oh, ow, out_c, out_h, out_w);
                    output->data[out_idx] = sum;
                }
            }
        }
    }
}

void conv2d_forward_layer(Layer* layer, Tensor* input, Tensor* output) {
    conv2d_forward(input, layer->weights, layer->bias, output, layer->stride, layer->padding);
}

void conv2d_backward_layer(Layer* layer, Tensor* d_out, Tensor* d_input) {
    // stub for now
}
