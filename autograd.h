#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include "matrix_math.h"
#include "matrix.h"
#include <string.h>

#ifndef GRAPH_POPULATION
#define GRAPH_POPULATION 32
#endif
void matrix_zero_grad(matrix* out);
void matrix_grad(matrix* out);
void matrix_backward(matrix* out);



#endif /* ifndef AUTOGRAD_H
 */
