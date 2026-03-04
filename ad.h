//	A value valued autodifferentiation library
#ifndef EGAD_H
#define EGAD_H 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h> 
#define NUM_PREVS 2
#define NEXT 2
#define GRAPH_SIZE 256
#define GRAPH_EQUALITY(inp1, inp2) (inp1->tape == inp2->tape) 

struct value; 
struct graph; 
typedef struct value value; 

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

struct value {
	OPTYPE op;
	double data;
	double grad;
	int* ref_count;
	struct value* previous[NUM_PREVS]; 
} ;


const char* get_optype_string(OPTYPE op);
value* value_init	(double data);
void value_print	(value* val);
void value_free		(value* val);

value* value_add	(value* inp1, value* inp2);
value* value_sub	(value* inp1, value* inp2);
value* value_mul	(value* inp1, value* inp2);
value* value_pow	(value* inp1, value* exponent);
value* value_sigmoid(value* inp1);
value* value_tanh	(value* inp1);
value* value_log	(value* inp1);
value* value_exp	(value* inp1);
value* value_sin	(value* inp1);
value* value_cos	(value* inp1);
value* value_relu	(value* inp1);
bool value_equality(value* inp1, value* inp2);
void value_backward(value* out);



#endif /* ifndef EGAD_H */
