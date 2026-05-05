#include <stdio.h>
#include <math.h>
#include "xgboost.h"

int main() {
    // Input vector set
    float inputs[5][10] = {
        {0.9648f, -0.0664f, 0.9868f, -0.3581f, 0.9973f, 1.1819f, -1.6157f, -1.2102f, -0.6281f, 1.2273f},
        {-0.9165f, -0.5664f, -1.0086f, 0.8316f, -1.1770f, 1.8205f, 1.7524f, -0.9845f, 0.3639f, 0.2095f},
        {-0.1095f, -0.4328f, -0.4576f, 0.7938f, -0.2686f, -1.8364f, 1.2391f, -0.2464f, -1.0581f, -0.2974f},
        {1.7504f, 2.0236f, 1.6882f, 0.0068f, -1.6077f, 0.1847f, -2.6194f, -0.3574f, -1.4731f, -0.1900f},
        {-0.2247f, -0.7113f, -0.2208f, 0.1171f, 1.5361f, 0.5975f, 0.3486f, -0.9392f, 0.1759f, 0.2362f}
    };

    // Python predictions
    float python_preds[5] = {
        0.3064f, 0.6952f, 0.6932f, 0.3141f, 0.6907f
    };

    // Test loop
    for(int i=0; i<5; i++) {
        printf("Test sample %d:\n", i+1);
        float output;
        output = predict(inputs[i]);
        printf("C prediction: %.4f\n", output);
        printf("Python prediction: %.4f\n", python_preds[i]);
        printf("Match: %s\n", fabs(output - python_preds[i]) < 0.01f ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}
