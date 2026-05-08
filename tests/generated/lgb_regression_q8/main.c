#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float result = predict(x);
    printf("predict: %f\n", result);
    (void)result;
    return 0;
}
