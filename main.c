#include <stdio.h>
#include "conv2d.h"
#include "tensor.h"

int main() {
    // Input: 1×1×4×4 
    Tensor input = tensor_alloc(1, 1, 4, 4);
    for (int i = 0; i < 16; i++) {
        input.data[i] = i + 1; // Fill with 1 to 16
    }

    // Weights: 1×1×3×3
    Tensor weights = tensor_alloc(1, 1, 3, 3);
    float kernel_vals[9] = {
         1,  0, -1,
         1,  0, -1,
         1,  0, -1
    };
    for (int i = 0; i < 9; i++) {
        weights.data[i] = kernel_vals[i];
    }

    // Bias: 1 value
    Tensor bias = tensor_alloc(1, 1, 1, 1);
    bias.data[0] = 0.0f;

    // Output: 1×1×2×2 (from (4-3)/1 + 1 = 2)
    Tensor output = tensor_alloc(1, 1, 2, 2);

    // Run convolution
    conv2d_forward(&input, &weights, &bias, &output, 1, 0);

    // Print output
    printf("Output:\n");
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
            int idx = IDX4(0, 0, h, w, 1, 2, 2);
            printf("%6.2f ", output.data[idx]);
        }
        printf("\n");
    }

    // Clean up
    tensor_free(input);
    tensor_free(weights);
    tensor_free(bias);
    tensor_free(output);

    return 0;
}
