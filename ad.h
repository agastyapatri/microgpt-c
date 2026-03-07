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
void ad_backward (ad_value* out);


typedef struct ad_matrix{
	uint rows; 			// rows  
	uint cols;  		// cols 
	uint size;
	ad_value* data;		// flat representation; row major order.	
} ad_matrix;
#define offset(m, i, j) (i*m->cols + j)
#define rand_double() (rand()/(double)RAND_MAX);

ad_matrix* ad_matrix_alloc(uint nrows, uint ncols);
void ad_matrix_free(ad_matrix* m);
void ad_matrix_print(ad_matrix* m);

//	creates a matrix of ad_values initialized to a gaussian distribution
ad_matrix* ad_matrix_random_normal(int nrows, int ncols, double mu, double sigma);










#endif /* ifndef EGAD_H */
