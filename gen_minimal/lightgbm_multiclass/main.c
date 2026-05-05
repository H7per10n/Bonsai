#include <stdio.h>
#include <math.h>
#include "lightgbm.h"

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
        {0.1494f, 0.2161f, 0.2391f, 0.3955f},
        {0.1875f, 0.1904f, 0.4214f, 0.2007f},
        {0.1881f, 0.1577f, 0.2701f, 0.3841f},
        {0.4953f, 0.1668f, 0.1476f, 0.1902f},
        {0.4875f, 0.1842f, 0.1639f, 0.1644f}
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
