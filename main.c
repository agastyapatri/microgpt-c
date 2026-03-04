#include "ad.h"
// #include <stdio.h>
// #include <stdlib.h>
int main(void){
	value* v1 = value_init(10.0);
	value* v2 = value_init(12.0);
	value* v3 = value_sin(v1);
	value* v4 = value_cos(v2);
	value* v5 = value_sigmoid(v3);
	value* v6 = value_init(13.0);
	value* v7 = value_exp(v5);
	value* v8 = value_mul(v6, v7);
	value* v9 = value_pow(v8, v4);
	value_backward(v9);
}
