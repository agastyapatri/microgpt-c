/*
 *	A library of commonly used math routines and constants
 */ 
#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H


#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include "matrix.h"

#define PI 3.1415926545897932	
#define SQRT2 1.414213562373
#define LN2 0.69314718056
#define EPSILON (double)1e-9	

typedef double (*unary_op)(double);
typedef double (*binary_op)(double, double);
typedef double (*ternary_op)(double, double, double);


static inline void MATRIX_UNARY_OP(matrix* inp1, matrix* out, unary_op function){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i]);
}
static inline void MATRIX_BINARY_OP(matrix* inp1, matrix* inp2, matrix* out, binary_op function){
	for(size_t i = 0; i < inp1->size; i++)
		out->data[i] = function(inp1->data[i], inp2->data[i]);
}

static inline void BUF_MSE(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	*out = 0;
	for(size_t i = 0 ; i < rows; i++){
		double* inp1row = inp1  + (i * stride);
		double* inp2row = inp2  + (i * stride);
		for(size_t j = 0; j < cols; j++){
			*out += pow((inp1row[j] - inp2row[j]), 2);
		}
	}
	*out /= (rows * cols);
}

static inline void BUF_MAE(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	*out = 0;
	for(size_t i = 0 ; i < rows; i++){
		double* inp1row = inp1  + (i * stride);
		double* inp2row = inp2  + (i * stride);
		for(size_t j = 0; j < cols; j++){
			*out += fabs(inp1row[j] - inp2row[j]);
		}
	}
	*out /= (rows * cols);
}

static inline void BUF_MATMUL(double* inp1, double* inp2, double* out, size_t inp1rows, size_t inp1cols, int inp1stride, size_t inp2cols, int inp2stride, int outstride){
	for(size_t i = 0; i < inp1rows; i++){
		double* inp1row = inp1 + (i * inp1stride);
		double* outrow = out + (i * outstride);
		for(size_t k = 0; k < inp1cols; k++){
			double inp1_ik = inp1row[k];
			double* inp2row = inp2 + (k * inp2stride);
			for(size_t j = 0; j < inp2cols; j++){
				outrow[j] += inp1_ik * inp2row[j];
			}
		} 
	}
} 

static inline void BUF_SUM(double* inp1, double* out, size_t rows, size_t cols, size_t stride){
	*out = 0; 
	for(size_t i = 0 ; i < rows; i++){
		double* inp1row = inp1 + (i * stride);
		for(size_t j = 0 ; j < cols; j++){
			*out += inp1row[j];
		} 
	}
}

static inline void BUF_MEAN(double* inp1, double* out, size_t rows, size_t cols, size_t stride){
	*out = 0; 
	for(size_t i = 0 ; i < rows; i++){
		double* inp1row = inp1 + (i * stride);
		for(size_t j = 0 ; j < cols; j++){
			*out += inp1row[j];
		} 
	}
	*out /=  (rows * cols);
}

static inline void BUF_STD(double* inp1, double* std, size_t rows, size_t cols, size_t stride){
	*std = 0; 
	double mean = 0;
	BUF_MEAN(inp1, &mean, rows, cols, stride);
	for(size_t i = 0; i < rows; i++){
		double* inp1row = inp1 + (i * stride);
		for(size_t j = 0; j < cols; j++){
			*std += (inp1row[j] - mean) * (inp1row[j] - mean);
		} 

	}
	*std /= (rows * cols);
	*std = sqrt(*std);
}


static inline void BUF_ADD(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_add_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}

static inline void BUF_POW(double* inp1, double* inp2, double* out, size_t size){
	for(size_t i = 0; i < size; i++){
		out[i] = pow(inp1[i], inp2[i]);
	}
}


static inline void BUF_SIN(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = sin(inprow[j]);
		}
	}
}

static inline void BUF_TANH(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = tanh(inprow[j]);
		}
	}
}


static inline void BUF_COS(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = cos(inprow[j]);
		}
	}
}

static inline void BUF_SIGMOID(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = 1.0 / (1 + exp(-inprow[j]));
		}
	}
}

static inline void BUF_RELU(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = (inprow[j] > 0) ? inprow[j] : 0;
		}
	}
}


static inline void BUF_LOG(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = log(inprow[j]);
		}
	}
}

static inline void BUF_EXP(double* inp, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double* inprow = inp + (i * stride);
		double* outrow = out + (i * stride);
		for(size_t j = 0; j < cols; j++){
			outrow[j] = exp(inprow[j]);
		}
	}
}


static inline void BUF_SOFTMAX(double* inp1, double* out, size_t rows, size_t cols, size_t stride){
	for(size_t i = 0; i < rows; i++){
		double rowsum = 0.0; 
		for(size_t j = 0; j < cols; j++){
			rowsum += exp(*(inp1 + (i*stride + j)));
		}
		for(size_t j = 0; j < cols; j++){
			*(out + (i * stride + j)) = exp(*(inp1 + (i*stride + j))) / rowsum; 
		} 
	}
}

static inline void BUF_SUB(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_sub_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] - d2[j];
		}
	}
}

static inline void BUF_MUL(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_mul_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] * d2[j];
		}
	}
}

static inline void BUF_DIV(double* inp1, double* inp2, double* out, size_t rows, size_t cols, size_t stride){
	size_t vector_limit = (cols / 4) * 4;
	for(size_t i = 0; i < rows; i++){
		double* d1 = inp1 + (i * stride);
		double* d2 = inp2 + (i * stride);
		double* o = out + (i * stride);
		size_t j = 0;
		for(; j <= vector_limit; j+=4){
			__m256d v1 = _mm256_load_pd(&d1[j]);
			__m256d v2 = _mm256_load_pd(&d2[j]);
			__m256d res = _mm256_div_pd(v1, v2);
			_mm256_store_pd(&o[j], res);
		}
		for(; j < cols; j++){
			o[j] = d1[j] + d2[j];
		}
	}
}
/*********************************************
 *	SCALAR FUNCTIONS
 *********************************************/ 

static inline double dtanh(double x){
	return 1 - pow(tanh(x), 2);
}

static inline double dsquare(double x){
	return 2*x;
}

static inline double dcube(double x){
	return 3*x*x;
}

static inline double dexp(double x){
	return exp(x);
}

static inline double dlog(double x){
	return 1 / x;
}


static inline double dsin(double x){
	return cos(x);
}

static inline double dcos(double x){
	return -sin(x);
}

static inline double dtan(double x){
	return 1 + pow(tan(x), 2);
}

static inline double dsigmoid(double x){
	return (1.0 / (1 + exp(-x)))*(1 - (1.0 / (1 + exp(-x)))	);
}

static inline double drelu(double x){
	return (x > 0) ? 1 : 0;

}
static inline double rand_double(){
	return rand()/(double)RAND_MAX;
}


static inline double normal(double x, double mu, double sigma){
	double temp = ((x - mu)*(x - mu))/(sigma*sigma);
	return (1/sqrt(2 * PI * sigma))*(exp(-0.5 * temp));
}


static inline double rand_normal(double mu, double sigma){
	double n2 = 0.0; 
	double n2_cached = 0.0; 
	if(!n2_cached){
		double u1 = rand_double();
		double u2 = rand_double();
		double r = sqrt(-2.0 * log(u1));
		double theta = 2 * PI * u2;
		n2 = r * sin(theta);
		n2_cached = 1;
		return r * cos(theta) * sigma + mu;
	}
	else{
		n2_cached = 0;
		return n2*sigma + mu;
	}

}


static inline double rand_uniform(double left, double right){
	return rand_double()*(right - left) + left;
}


static inline double squared_error(double x, double y){
	return (x - y)*(x - y);
}


static inline char* get_optype_string(OPTYPE op){
	switch (op) {
		case MSE: 
			return "mse";
		case MAE: 
			return "mae";
		case SUM: 
			return "sum";
		case STD: 
			return "std";
		case MEAN: 
			return "mean";
		case ADD: 
			return "add";
		case POW: 
			return "pow";
		case SUB: 
			return "sub";
		case MUL: 
			return "mul";
		case NONE: 
			return "none";
		case SIN:
			return "sin";
		case COS:
			return "cos";
		case LOG: 
			return "log";
		case EXP: 
			return "exp";
		case MATMUL: 
			return "matmul";
		case TANH: 
			return "tanh";
		case SIGMOID: 
			return "sigmoid";
		case SOFTMAX: 
			return "softmax";
		case RELU: 
			return "relu";
	}
	return NULL;
}

#endif // !MATRIX_MATH_H
