// #define PLOT 1
#ifdef PLOT 
#define _POSIX_C_SOURCE 200809L
#endif 

#include <assert.h>
#include "matrix.h"
#include <math.h>
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "matrix_math.h"

matrix* matrix_alloc(int ROWS, int COLS, bool requires_grad){
	matrix* m = malloc(sizeof(matrix));
	if(!m){
		return NULL; 
	}
	m->rows = ROWS;
	m->cols = COLS;
	m->op = NONE;
	m->ref_count = 1;
	m->stride = (m->cols + (ALIGNMENT / sizeof(double)) - 1)  & ~((ALIGNMENT / sizeof(double)) - 1);
	m->padding = m->stride - m->cols;
	m->size = m->rows * m->stride;
	m->bytes = m->size * sizeof(double);
	m->data = (double*)aligned_alloc(ALIGNMENT, m->bytes);
	if(!(m->data)){
		free(m);
		return NULL; 
	}
	memset(m->data, 0, m->bytes);

	m->num_prevs = 0; 
	m->previous[0] = NULL;
	m->previous[1] = NULL;
	m->grad = NULL;
	m->requires_grad = requires_grad;
	if(m->requires_grad){
		m->grad = (double*)aligned_alloc(ALIGNMENT, m->bytes);
		if(!(m->grad)){
			free(m->data);
			free(m);
			return NULL; 
		}
		memset(m->grad, 0, m->bytes);
	}
	return m;
}


void matrix_grad_on(matrix* m){
	m->requires_grad = true;
	size_t bytes = sizeof(double)*m->size;
	bytes += ALIGNMENT - (bytes % ALIGNMENT);
	m->grad = (double*)aligned_alloc(ALIGNMENT, bytes);
	for(size_t i = 0; i < m->size; i++)
		m->grad[i] = 0.0;
}

void matrix_grad_off(matrix* m){
	m->requires_grad = 0; 
	m->op = NONE;
	if(m->grad){
		free(m->grad);
	}
}


matrix* matrix_ones(int ROWS, int COLS, bool requires_grad){
	matrix* m = matrix_alloc(ROWS,  COLS, requires_grad);
	for(size_t i = 0; i < m->rows; i++){
		double* mdatarow = m->data + (i * m->stride);
		for(size_t j = 0; j < m->cols; j++){
			mdatarow[j] = 1;
		} 
	}
	return m;
}

matrix* matrix_zeros(int ROWS, int COLS, bool requires_grad){
	matrix* out = matrix_alloc(ROWS, COLS, requires_grad);
	return out;
}

matrix* matrix_eye(int SIDE, bool requires_grad){
	matrix* m = matrix_alloc(SIDE, SIDE, requires_grad);
	for(size_t i = 0; i < m->rows; i++){
		for(size_t j = 0; j < m->cols; j++){
			m->data[offset(m, i, j)] = (i == j ) ? 1 : 0;
		} 
	}
	return m;
}

matrix* matrix_transpose(matrix* m){
	matrix* out = matrix_alloc(m->cols, m->rows, false);
	for(size_t i = 0; i < m->rows; i++){
		for(size_t j = 0; j < m->cols; j++){
			out->data[offset(out, j, i)] = m->data[offset(m, i, j)];
		} 
	}
	return out;
}

void matrix_print_shape(matrix* m){
	printf("(%li, %li)\n", m->rows, m->cols);
}

void matrix_free(matrix* m){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("ERROR: argument in matrix_free() is NULL\n");
	}
	if(m->ref_count == 0){
		MATRIX_ERROR("ERROR: argument in matrix_free() is already freed.\n");
	}
	m->ref_count--;
	if(m->ref_count == 0){
		free(m->data);
		if(!m->requires_grad){
			free(m->grad);
		}
	}
	free(m);
}

void matrix_print(matrix *m){
	if(MATRIX_NULL(m)){
		MATRIX_ERROR("ERROR: argument in print_matrix() is NULL\n");
	}
	printf("matrix([");

	for(size_t i = 0; i < m->rows; i++){
		double* row = m->data + i*m->stride;
		if(i >= 1) printf("        ");
		printf("[");
		for(size_t j = 0; j < m->cols; j++){
			printf("%10.7f", row[j]);
			if(j != m->cols-1) printf(", ");
		}
		printf("]");
		if(!(i == m->rows-1)) printf(",\n");

	}

	char* opstring = get_optype_string(m->op);
	printf("]");
	if(m->requires_grad){
		printf(", requires_grad = %d, optype = %s", m->requires_grad, opstring);
	}
	printf(")\n");
}



matrix* matrix_matmul(matrix* inp1, matrix* inp2){
	assert(inp1 != NULL && inp2 != NULL);
	assert(inp1->cols == inp2->rows);
	matrix* out = matrix_alloc(inp1->rows, inp2->cols, inp1->requires_grad || inp2->requires_grad);
	BUF_MATMUL(inp1->data, inp2->data, out->data, inp1->rows, inp1->cols, inp1->stride, inp2->stride, inp2->stride, out->stride);
	if(out->requires_grad){
		out->op = MATMUL;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		inp1->ref_count++;
		inp2->ref_count++;
	}
	return out;
}


matrix* matrix_add(matrix* inp1, matrix* inp2){
	assert(inp1 != NULL && inp2 != NULL);
	assert((inp1->rows == inp2->rows) && (inp1->cols == inp2->cols));

	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	BUF_ADD(inp1->data, inp2->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = ADD;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		inp1->ref_count++;
		inp2->ref_count++;
	}
	return out;
}
matrix* matrix_pow(matrix* inp1, matrix* inp2){
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	BUF_POW(inp1->data, inp2->data, out->data, inp1->size);
	if(out->requires_grad){
		out->op = POW;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		inp1->ref_count++;
		inp2->ref_count++;
	}
	return out;
}

matrix* matrix_sin(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_SIN(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = SIN;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_sigmoid(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_SIGMOID(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = SIGMOID;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_relu(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_RELU(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = RELU;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_tanh(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_TANH(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = TANH;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}


matrix* matrix_softmax(matrix* inp1){
	assert(!MATRIX_NULL(inp1));
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad);
	BUF_SOFTMAX(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = SOFTMAX;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_rmsnorm(matrix* inp1){
	assert(!MATRIX_NULL(inp1));
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad);
	BUF_RMSNORM(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = RMSNORM;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_log(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_LOG(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = LOG;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}
matrix* matrix_exp(matrix* inp1){
	assert(inp1 != NULL);
	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_EXP(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = EXP;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_cos(matrix* inp1){
	assert(inp1 != NULL);

	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad );
	BUF_COS(inp1->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	if(out->requires_grad){
		out->op = COS;
		out->previous[0] = inp1;
		out->previous[1] = NULL;
		out->num_prevs = 1;
		inp1->ref_count++;
	}
	return out;
}

matrix* matrix_sub(matrix* inp1, matrix* inp2){
	assert(inp1 != NULL && inp2 != NULL);
	assert((inp1->rows == inp2->rows) && (inp1->cols == inp2->cols));

	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	BUF_SUB(inp1->data, inp2->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = SUB;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		inp1->ref_count++;
		inp2->ref_count++;
	}
	return out;
} 

matrix* matrix_mul(matrix* inp1, matrix* inp2){
	assert(inp1 != NULL && inp2 != NULL);
	assert((inp1->rows == inp2->rows) && (inp1->cols == inp2->cols));

	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
	BUF_MUL(inp1->data, inp2->data, out->data, inp1->rows, inp1->cols, inp1->stride);
	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
	if(out->requires_grad){
		out->op = MUL;
		out->previous[0] = inp1;
		out->previous[1] = inp2;
		out->num_prevs = 2;
		inp1->ref_count++;
		inp2->ref_count++;
	}
	return out;
}

// matrix* matrix_div(matrix* inp1, matrix* inp2){
// 	matrix* out = matrix_alloc(inp1->rows, inp1->cols, inp1->requires_grad || inp2->requires_grad);
// 	MATRIX_DIV(inp1, inp2, out);
// 	out->requires_grad = inp1->requires_grad || inp2->requires_grad;
// 	if(out->requires_grad){
// 		out->op = DIV;
// 		out->previous[0] = inp1;
// 		out->previous[1] = inp2;
// 		out->num_prevs = 2;
// 		(*(inp1->ref_count))++;
// 		(*(inp2->ref_count))++;
// 	}
// 	return out;
// }


matrix* matrix_reshape(matrix* m, size_t ROWS, size_t COLS){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL argument passed to matrix_reshape()\n");
	if(m->rows*m->cols != ROWS*COLS)
		MATRIX_ERROR("Invalid ROWS or COLS argument(s) passed to matrix_reshape(); Ensure that the cardinality of the intended matrix is the same as the target matrix\n");
	matrix* out = (matrix*)malloc(sizeof(matrix));
	out->rows = ROWS;
	out->cols = COLS;
	out->size = m->size;
	out->data = m->data;
	out->ref_count = m->ref_count;
	out->requires_grad = m->requires_grad;
	out->grad = m->grad;
	m->ref_count++;
	return out;
}

void matrix_scale(matrix* a, double b){
	for(size_t i = 0; i < a->size; i++){
		a->data[i] = b*a->data[i];
	}
} 


bool matrix_equality(matrix* a, matrix* b){
	assert((a->rows == b->rows)); 
	assert((a->cols == b->cols));
	for(size_t i = 0; i < a->rows; i++){
		double* arow = a->data + (i * a->stride);
		double* brow = b->data + (i * b->stride);
		for(size_t j = 0; j < a->cols; j++){
			if(!((arow[j] >= brow[j] - EPSILON) && (arow[j] <= brow[j] + EPSILON))){
				return false;
			}
		}
	}
	return true;

}

void matrix_randomize(matrix* m, double (*function)(double mu, double sigma)){
	for(size_t i = 0; i < m->size; i++){
		m->data[i] = function(0, 1);
	}
}

matrix* matrix_random_uniform(int ROWS, int COLS, double left, double right, bool requires_grad){
	matrix* m = matrix_alloc( ROWS,  COLS, requires_grad);
	for(int i = 0; i < ROWS; i++){
		double* row = m->data + (i * m->stride);
		for(int j = 0; j < COLS; j++){
			row[j] = rand_uniform(left, right);
		} 
	}
	return m;
}


matrix* matrix_random_normal(int ROWS, int COLS, double mu, double sigma, bool requires_grad){
	assert(sigma > (0 + 100*EPSILON));
	matrix* m = matrix_alloc(ROWS, COLS, requires_grad);
	for(int i = 0; i < ROWS; i++){
		double* row = m->data + (i * m->stride);
		for(int j = 0; j < COLS; j++){
			row[j] = rand_normal(mu, sigma);
		} 
	}
	return m;
}


matrix* matrix_add_rowwise(matrix* mat, matrix* vec){
	assert(vec->rows == 1);
	assert(mat->cols == vec->cols);
	assert(!(MATRIX_NULL(mat) || MATRIX_NULL(vec)));
	matrix* out = matrix_alloc(mat->rows, mat->cols, mat->requires_grad || vec->requires_grad);
	double* vecrow = vec->data;
	for(size_t i = 0; i < out->rows; i++){
		double* outrow = out->data + (i * out->stride);
		double* matrow = mat->data + (i * mat->stride);
		for(size_t j = 0; j < out->cols; j++){
			outrow[j] = matrow[j] + vecrow[j];
		}
	}
	if(out->requires_grad){
		out->num_prevs = 2;
		out->previous[0] = (matrix*)mat; 
		out->previous[1] = (matrix*)vec;
		out->op = ADD; 
		mat->ref_count++;
		vec->ref_count++;
	}
	return out;
}



void matrix_scalar_mul(matrix* input, double scalar, matrix* output){
	for(size_t i = 0; i < input->size; i++){
		output->data[i] = scalar*input->data[i];
	}
}

matrix* matrix_copy(const matrix* input){
	if(MATRIX_NULL(input)){
		MATRIX_ERROR("Invalid matrix argument(s) in matrix_copy()\n");
	}
	matrix* output = matrix_alloc(input->rows, input->cols, input->requires_grad);
	for(size_t i = 0; i < output->size; i++){
		output->data[i] = input->data[i];
	}
	if(input->requires_grad){
		matrix_grad_on(output);
	}
	return output;
}



matrix* matrix_max(const matrix* m){
	if(MATRIX_NULL(m) || m->ref_count == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	matrix* max = matrix_alloc(1,1, 0); 
	max->data[0] = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max->data[0] <= m->data[i]) 
			max->data[0] = m->data[i];
	}
	return max;
}

matrix* matrix_min(const matrix* m){
	if(MATRIX_NULL(m) || m->ref_count == 0){
		MATRIX_ERROR("matrix passed to matrix_max() is NULL or already has been freed.\n");
	}
	matrix* max = matrix_alloc(1,1, 0); 
	max->data[0] = m->data[0];
	for(size_t i = 1; i < m->size; i++){
		if(max->data[0] > m->data[i]) 
			max->data[0] = m->data[i];
	}
	return max;
}

matrix* matrix_mean(matrix* m){
	assert(!MATRIX_NULL(m));
	matrix* mean = matrix_alloc(1, 1, m->requires_grad);
	BUF_MEAN(m->data, mean->data, m->rows, m->cols, m->stride);
	if( mean->requires_grad){
		mean->op = MEAN;
		mean->previous[0] = m;
		mean->num_prevs = 1;
	}
	return mean;
}


matrix* matrix_std(matrix* m){
	assert(!MATRIX_NULL(m));
	matrix* std = matrix_alloc(1, 1, m->requires_grad);
	BUF_STD(m->data, std->data, m->rows, m->cols, m->stride);
	if( std->requires_grad){
		std->op = STD;
		std->previous[0] = m;
		std->num_prevs = 1;
	}
	return std;
}

matrix* matrix_sum(matrix* m){
	assert(!MATRIX_NULL(m));
	matrix* sum = matrix_alloc(1, 1, m->requires_grad);
	BUF_SUM(m->data, sum->data, m->rows, m->cols, m->stride);
	if(sum->requires_grad){
		sum->op = SUM;
		sum->previous[0] = m;
		sum->num_prevs = 1;
	}
	return sum;
}


matrix* matrix_mse(matrix* inp1, matrix* inp2){
	assert((inp1->rows == inp2->rows) || (inp1->cols == inp2->cols));
	assert(!(MATRIX_NULL(inp1) || MATRIX_NULL(inp2)));
	matrix* mse = matrix_alloc(1, 1, inp1->requires_grad || inp2->requires_grad);
	BUF_MSE(inp1->data, inp2->data, mse->data, inp1->rows, inp1->cols, inp1->stride);
	if(mse->requires_grad){
		mse->op = MSE; 
		mse->num_prevs = 1; 
		mse->previous[0] = inp1;
		inp1->ref_count++;
	}
	return mse;
}

matrix* matrix_mae(matrix* inp1,const matrix* inp2){
	assert((inp1->rows == inp2->rows) || (inp1->cols == inp2->cols));
	assert(!(MATRIX_NULL(inp1) || MATRIX_NULL(inp2)));
	matrix* mse = matrix_alloc(1, 1, inp1->requires_grad || inp2->requires_grad);
	BUF_MAE(inp1->data, inp2->data, mse->data, inp1->rows, inp1->cols, inp1->stride);
	if(mse->requires_grad){
		mse->op = MAE; 
		mse->num_prevs = 1; 
		mse->previous[0] = inp1;
		inp1->ref_count++;
	}
	return mse;
}

matrix* matrix_linspace(double start, double end, size_t num, bool requires_grad){
	if(end < start){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start\n");
	}
	matrix* out = matrix_alloc(1, num, requires_grad);
	double step = (end - start)/num;
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}

matrix* matrix_arange(double start, double end, double step, bool requires_grad){
	if((end < start) || step <= 0){
		MATRIX_ERROR("Invalid argument(s) in matrix_linspace(); Ensure end > start and step >= 0\n");
	}
	size_t num = (end - start)/step;
	matrix* out = matrix_alloc(1, num, requires_grad);
	for(size_t i = 0; i < num; i++){
		out->data[i] = start + i*step; 
	}
	return out;
}

// TODO
double matrix_det(const matrix* m){
	if(MATRIX_NULL(m))
		MATRIX_ERROR("NULL matrix in matrix_determinant()\n");
	if(m->rows == 2 && m->cols == 2){
		return (get(m, 0, 0)*get(m, 1, 1))-(get(m, 0, 1)*get(m, 1, 0));
	}
	return 0;
}


//	TODO
matrix* matrix_inverse(const matrix* m){
	matrix* out = matrix_alloc(m->rows, m->cols, m->requires_grad);
	if(m->rows == 2 && m->cols == 2){
		set(out, (1/matrix_det(m))*(get(m, 1, 1)), 0, 0) ; 
		set(out, (1/matrix_det(m))*(-get(m, 0, 1)), 0, 1) ; 
		set(out, (1/matrix_det(m))*(-get(m, 1, 0)), 1, 0) ; 
		set(out, (1/matrix_det(m))*(get(m, 0, 0)), 1, 1) ; 
	}
	return out;
}

double matrix_trace(const matrix* m){
	if(!matrix_is_square(m))
		MATRIX_ERROR("Matrix argument is not square in matrix_trace()\n");
	double trace = 0;
	for(size_t i = 0; i < m->rows; i++){
		trace += get(m, i, i);
	}
	return trace;
}

matrix* matrix_from_raw(double* arr, size_t rows, size_t cols){
	matrix* out = (matrix*)malloc(sizeof(matrix));
	out->requires_grad = 0;
	out->rows = rows; 
	out->cols = cols;
	out->op = NONE;
	out->stride = (out->cols + (ALIGNMENT / sizeof(double)) - 1)  & ~((ALIGNMENT / sizeof(double)) - 1);
	out->padding = out->stride - out->cols;
	out->size = out->rows * out->stride;
	out->bytes = out->size * sizeof(double);
	out->data = arr;
	out->ref_count = 1;
	out->grad = NULL;
	return out;
}

matrix* matrix_get_row(matrix* inp1, size_t row_idx){
	matrix* out = matrix_alloc(1, inp1->cols, inp1->requires_grad);
	for(size_t i = 0; i < out->cols; i++)
		out->data[offset(out, 0, i)] = inp1->data[offset(inp1, row_idx, i)];
	return out;
}






















#ifdef PLOT 
#define _POSIX_C_SOURCE 200809L
void plot(matrix* x, matrix* y){
	int len = x->rows * x->cols;
	FILE *gnuplotpipe = popen("gnuplot -persistent", "w");
	if(!gnuplotpipe){
		fprintf(stderr, "ERROR: Could not open GNUPLOT pipe\n");
		exit(1);
	}
	fprintf(gnuplotpipe, "set title 'Plotting a Matrix'\n");
	fprintf(gnuplotpipe, "set xlabel 'X'\n");
	fprintf(gnuplotpipe, "set ylabel 'f(X)'\n");
	fprintf(gnuplotpipe, "set grid\n");
	fprintf(gnuplotpipe, "plot '-'  with points title 'f(x)'\n");
	for(int i = 0; i < len; i++){
		fprintf(gnuplotpipe, "%lf %lf\n", x->data[i], y->data[i]);
		fprintf(gnuplotpipe, "%lf %lf\n", x->data[i], x->grad[i]);
	}
	fprintf(gnuplotpipe, "e\n");
	fflush(gnuplotpipe);
	pclose(gnuplotpipe);
}
#endif
