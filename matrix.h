#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_PREVS 3 
#define MAX_ARGS 5 
#define MAX_PARAM_MATRICES 10

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ALIGNMENT
#define ALIGNMENT 32
#endif

#define MATRIX_TIMER(function) clock_t start = clock(); \
						function;						\
						clock_t end = clock();  		\
						printf("%0.10f\n", (double)(end - start)/CLOCKS_PER_SEC);

#define MATRIX_NULL(m) (m==NULL) ? 1 : 0
#define MATRIX_ERROR(msg) printf(msg);	\
						  exit(EXIT_FAILURE);
#define IN_COMP_GRAPH(inp1, inp2) inp1->requires_grad || inp2->requires_grad

typedef enum {
	NONE,
	ADD, 
	SUB,
	MUL,
	MATMUL,
	SIN,
	COS,
	TANH,
	LOG, 
	EXP,
	POW,
	SIGMOID,
	SOFTMAX,
	RELU,
	SUM,
	MEAN,
	STD,
	MSE,
	MAE,
	// ARCSIN,
	// ARCCOS,
	// ARCTAN,
	// SINH,
	// COSH,
	// SQUARE,
	// CUBE,
	// DIV
} OPTYPE;


typedef struct matrix{
	size_t 	rows;
	size_t 	cols;
	int 	ref_count;
	double* data; 
	size_t 	bytes;
	int 	stride;
	size_t 	size;
	int 	padding;
	bool 	requires_grad;
	double* grad;
	OPTYPE 	op;
	int 	num_prevs;
	struct matrix* previous[MAX_PREVS];
} matrix;


static inline size_t offset(const matrix* m, int i, int j){
	return (i*m->stride + j);
}
static inline double get(const matrix* m, int i, int j){
	return m->data[offset(m, i, j)];
}
static inline void set(matrix* m, double val, int i, int j){
	m->data[i*m->stride + j] = val;
}
static inline bool matrix_is_square(const matrix* m){
	return (m->rows == m->cols) ? 1 : 0;
}
static inline bool matrix_shape_equality(matrix* a, matrix* b){
	if(a->cols != b->cols || a->rows != b->rows){
		return false;
	}
	return true;
}


void 	matrix_print_shape(matrix* m);
matrix* matrix_max(const matrix* m);
matrix* matrix_min(const matrix* m);
matrix* matrix_mean(matrix* m);
matrix* matrix_std(matrix* m);
matrix* matrix_sum(matrix* m);
void 	matrix_grad_on(matrix* m);
void 	matrix_grad_off(matrix* m);
matrix* matrix_alloc(int ROWS, int COLS, bool requires_grad);
matrix* matrix_ones(int ROWS, int COLS, bool requires_grad);
matrix* matrix_zeros(int ROWS, int COLS, bool requires_grad);
matrix* matrix_eye(int SIDE, bool requires_grad);
matrix* matrix_linspace(double start, double end, size_t num, bool requires_grad);
matrix* matrix_arange(double start, double end, double step, bool requires_grad);
void 	matrix_print(matrix* m);
void 	matrix_free(matrix* m);
matrix* matrix_transpose(matrix* m);
matrix* matrix_copy(const matrix* input);
matrix* matrix_reshape(matrix* m, size_t ROWS, size_t COLS);
void 	matrix_scale(matrix* a, double b);
bool 	matrix_equality(matrix* a, matrix* b);
void 	matrix_randomize(matrix* m, double (*function)(double, double));
matrix* matrix_random_uniform(int ROWS, int COLS, double left, double right, bool requires_grad);
matrix* matrix_random_normal(int ROWS, int COLS, double mu, double sigma, bool requires_grad);
matrix* matrix_add_rowwise(matrix* mat, matrix* vec);
double  matrix_trace(const matrix* m);

//	mathematics for the comp graph construction
matrix* matrix_add(matrix* inp1, matrix* inp2);
matrix* matrix_sub(matrix* inp1, matrix* inp2);
matrix* matrix_mul(matrix* inp1, matrix* inp2);
matrix* matrix_div(matrix* inp1, matrix* inp2);
matrix* matrix_matmul(matrix* inp1, matrix* inp2);
matrix* matrix_pow(matrix* inp1, matrix* inp2);
matrix* matrix_sin(matrix* inp1);
matrix* matrix_cos(matrix* inp1);
matrix* matrix_log(matrix* inp1);
matrix* matrix_exp(matrix* inp1);
matrix* matrix_tanh(matrix* inp1);
matrix* matrix_softmax(matrix* inp1);
matrix* matrix_sigmoid(matrix* inp1);
matrix* matrix_relu(matrix* inp1);
matrix* matrix_mse(matrix* inp1, matrix* inp2);
matrix* matrix_mae(matrix* inp1, const matrix* inp2);

matrix* matrix_from_raw(double* arr, size_t rows, size_t cols);
//TODO 
void	matrix_push_back(matrix* mat, double* array);
matrix* matrix_sort(const matrix* m);
double 	matrix_det(const matrix* m);
matrix* matrix_inverse(const matrix* m);






#endif // !MATRIX_MATRIX_H
