#include <stdio.h>
#include <math.h>
#include "xgboost.h"

int main() {
    // Input vector set
    float inputs[5][15] = {
        {-0.2114f, 0.7888f, -0.9948f, -1.0269f, -1.3749f, 6.0690f, -0.2320f, -1.0431f, -1.1222f, -0.6190f, 3.0900f, 2.4441f, -0.1576f, -2.8292f, -2.6995f},
        {0.3599f, 1.9426f, -0.4753f, 0.3024f, -2.9625f, 2.1226f, 0.9040f, 1.8428f, -0.9674f, -0.8027f, -0.1508f, 1.9550f, 0.7687f, -0.7876f, -2.9437f},
        {-1.0139f, 1.2655f, 2.3846f, -2.1097f, 0.3796f, -0.3105f, 1.6091f, 0.9724f, -3.1764f, 1.0033f, -3.2815f, -0.1859f, 2.8515f, 3.1475f, 3.6595f},
        {-0.0828f, 0.5380f, 0.2948f, -3.1094f, 3.3196f, -2.8628f, -2.9599f, 1.5932f, -2.2279f, -0.1957f, 3.3193f, 0.6597f, -1.5485f, 1.0105f, 5.7709f},
        {-1.3230f, -0.0042f, 0.0960f, -0.0112f, 1.6045f, -5.3335f, -2.8627f, -0.5085f, 1.1500f, 0.2021f, -2.2954f, -0.3752f, -0.4669f, 1.3046f, 3.3092f}
    };

    // Python predictions
    float python_preds[5][4] = {
        {0.2075f, 0.2087f, 0.3121f, 0.2716f},
        {0.1591f, 0.2820f, 0.4004f, 0.1585f},
        {0.2067f, 0.2300f, 0.2247f, 0.3386f},
        {0.4595f, 0.1779f, 0.1763f, 0.1863f},
        {0.4386f, 0.2149f, 0.1727f, 0.1737f}
    };

    // Test loop
    for(int i=0; i<5; i++) {
        printf("Test sample %d:\n", i+1);
        float output[4];
        predict(inputs[i], output);
        printf("C prediction: "); for(int j=0; j<4; j++) printf("%.4f ", output[j]); printf("\n");
        printf("Python prediction: "); for(int j=0; j<4; j++) printf("%.4f ", python_preds[i][j]); printf("\n");
        
        int match = 1;
        for(int j=0; j<4; j++) {
            if(fabs(output[j] - python_preds[i][j]) > 0.01f) {
                match = 0;
                break;
            }
        }
        printf("Match: %s\n", match ? "Yes" : "No");

        printf("\n");
    }

    return 0;
}
