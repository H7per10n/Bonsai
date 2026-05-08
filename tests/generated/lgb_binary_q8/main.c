#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float prob = predict(x);
    int   cls  = predict_class(x);
    printf("predict: %f  class: %d\n", prob, cls);
    (void)prob; (void)cls;
    return 0;
}
