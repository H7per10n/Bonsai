#include <stdio.h>
#include <math.h>
#include "xgboost.h"

int main() {
    // Input vector set
    float inputs[5][8] = {
        {-1.4073f, -0.9561f, 2.3176f, 0.1255f, -1.2981f, -0.7249f, 0.0541f, 0.8100f},
        {0.0870f, -0.7735f, 1.8987f, 0.2988f, -1.0361f, -2.0858f, 0.9737f, -0.3432f},
        {1.3227f, -1.4962f, -0.9182f, 0.4567f, -0.1691f, 0.9104f, 1.5737f, -0.6052f},
        {-3.0076f, 1.0973f, -1.5656f, -0.4959f, 0.5712f, -2.4264f, -2.3869f, 1.2841f},
        {-0.3871f, -0.5496f, 0.8596f, -0.7667f, -0.0454f, 1.7194f, 1.9925f, -0.6059f}
    };

    // Python predictions
    float python_preds[5] = {
        -31.1431f, 21.8381f, 51.3320f, -162.3497f, 82.6116f
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
