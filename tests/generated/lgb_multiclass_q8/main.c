#include <stdio.h>
#include <math.h>
#include "model.h"

int main(void) {
    float x[N_FEATURES] = {0};
    float probs[N_CLASSES] = {0};
    predict(x, probs);
    int cls = predict_class(x);
    printf("class: %d\n", cls);
    (void)cls;
    return 0;
}
