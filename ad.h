//	A ad_value ad_valued autodifferentiation library
#ifndef EGAD_H
#define EGAD_H 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h> 
#define NUM_PREVS 2
#define NEXT 2
#define GRAPH_SIZE 256
#define PI 3.1415926545897932
#define MU 0 
#define SIGMA 0.08
#define GRAPH_EQUALITY(inp1, inp2) (inp1->tape == inp2->tape) 

typedef unsigned int uint;

typedef enum {
	NONE,
	ADD,
	SUB,
	MUL,
	POW,
	SIGMOID,
	TANH,
	RELU,
	SIN,
	COS,
	LOG, 
	EXP, 
} OPTYPE;

typedef struct ad_value {
	OPTYPE op;
	double data;
	double grad;
	int ref_count;
	struct ad_value* previous[NUM_PREVS]; 
} ad_value;


const char* get_optype_string(OPTYPE op);
ad_value* ad_value_random_gauss(double mu, double sigma);
ad_value* ad_value_alloc	(double data);
ad_value* ad_value_rand_normal(double mu, double sigma);
void ad_value_print		(ad_value* val);
void ad_value_free		(ad_value* val);

ad_value* ad_value_add	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_sub	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_mul	(ad_value* inp1, ad_value* inp2);
ad_value* ad_value_pow	(ad_value* inp1, ad_value* exponent);
ad_value* ad_value_sigmoid(ad_value* inp1);
ad_value* ad_value_tanh	(ad_value* inp1);
ad_value* ad_value_log	(ad_value* inp1);
ad_value* ad_value_exp	(ad_value* inp1);
ad_value* ad_value_sin	(ad_value* inp1);
ad_value* ad_value_cos	(ad_value* inp1);
ad_value* ad_value_relu	(ad_value* inp1);
bool ad_value_equality 	(ad_value* inp1, ad_value* inp2);
void ad_value_backward 	(ad_value* out);
double rand_normal(double mu, double sigma);


typedef struct ad_matrix{
	uint rows; 			// rows  
	uint cols;  		// cols 
	uint size;			// rows*cols
	ad_value* data;		// flat representation; ad_matrix.data is an array of ad_value.
} ad_matrix;
#define offset(m, i, j) (i*m->cols + j)
#define rand_double() (rand()/(double)RAND_MAX);
#define AD_MATRIX_RANDOM_INIT(matrix, size, mu, sigma)	for(uint j = 0; j < size; j++){						\
															matrix->data[j].data = rand_normal(mu, sigma);  \
															matrix->data[j].grad = 0; 						\
															matrix->data[j].op = NONE;  					\
															matrix->data[j].ref_count = 1; 					\
															matrix->data[j].previous[0] = NULL; 			\
															matrix->data[j].previous[1] = NULL; 			\
														}

#define AD_MATRIX_SHAPE_INIT(matrix, nrows, ncols)	matrix->rows = nrows;		 	\
													matrix->cols = ncols;		 	\
													matrix->size = nrows * ncols;	\

ad_matrix* ad_matrix_alloc(uint nrows, uint ncols);
void 	ad_matrix_free(ad_matrix* m);
void 	ad_matrix_print(ad_matrix* m);
double 	ad_matrix_mean(ad_matrix* m);
double 	ad_matrix_sum(ad_matrix* m);
double 	ad_matrix_std(ad_matrix* m);
double 	ad_matrix_max(ad_matrix* m);
double 	ad_matrix_min(ad_matrix* m);

//	creates a matrix of ad_values initialized to a gaussian distribution
ad_matrix* ad_matrix_random_normal(int nrows, int ncols, double mu, double sigma);
ad_matrix* ad_matrix_softmax(ad_matrix* x);
ad_matrix* ad_matrix_rmsnorm(ad_matrix* x);
ad_matrix* ad_matrix_matmul(ad_matrix* x, ad_matrix* y);

#endif /* ifndef EGAD_H */
