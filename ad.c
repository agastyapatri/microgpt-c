#include "ad.h"
#include <math.h>
#include <math.h>
#include <string.h>

const char* get_optype_string(OPTYPE op){
	switch (op) {
		case(ADD):
			return "add";
		case(SUB):
			return "sub";
		case(MUL):
			return "mul";
		case(POW):
			return "pow";
		case (SIGMOID):
			return "sigmoid";
		case (TANH):
			return "tanh";
		case (RELU):
			return "relu";
		case (NONE):
			return "none";
		case (SIN):
			return "sin";
		case (COS):
			return "cos";
		case (LOG):
			return "log";
		case (EXP):
			return "exp";
	} 
	return NULL;
}

value* value_init(double data){
	value* a = (value*)malloc(sizeof(value));
	if(!a)	return NULL;
	a->data = data; 
	a->grad = 0;
	a->op = NONE;
	a->previous[0] = NULL;
	a->previous[1] = NULL;
	a->ref_count = (int*)malloc(sizeof(int));
	if(!a){
		free(a);
		return NULL;
	}
	*(a->ref_count) = 1;
	return a;
}

void value_free(value* val){
	(*(val->ref_count))--;
	if(*(val->ref_count) == 0){
		free(val);
	}
}


void value_print(value* val){
	printf("value(data: %lf, grad: %lf, op: %s)", val->data, val->grad, get_optype_string(val->op));
}

value* value_add(value* inp1, value* inp2){
	value* out = value_init(inp1->data + inp2->data);
	out->op = ADD;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count[0]++;
	out->previous[1]->ref_count[0]++;

	return out;
}

value* value_sub(value* inp1, value* inp2){
	value* out = value_init(inp1->data - inp2->data);
	out->op = SUB;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count[0]++;
	out->previous[1]->ref_count[0]++;
	return out;
}
value* value_mul(value* inp1, value* inp2){
	value* out = value_init(inp1->data * inp2->data);
	out->op = MUL;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count[0]++;
	out->previous[1]->ref_count[0]++;
	return out;
}

value* value_pow(value* inp1, value* exponent){
	value* out = value_init(pow(inp1->data, exponent->data));
	out->op = POW;
	out->previous[0] = inp1;
	out->previous[1] = exponent;
	out->previous[0]->ref_count[0]++;
	out->previous[1]->ref_count[0]++;
	return out;
}

value* value_sigmoid(value* inp1){
	value* out = value_init(1.0 / (1 + exp(-(inp1->data))));
	out->op = SIGMOID;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}

value* value_tanh(value* inp1){
	value* out = value_init(tanh(inp1->data));
	out->op = TANH;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}
value* value_log(value* inp1){
	value* out = value_init(log(inp1->data));
	out->op = LOG;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}
value* value_exp(value* inp1){
	value* out = value_init(exp(inp1->data));
	out->op = EXP;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}

value* value_sin(value* inp1){
	value* out = value_init(sin(inp1->data));
	out->op = SIN;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}

value* value_cos(value* inp1){
	value* out = value_init(cos(inp1->data));
	out->op = COS;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}


value* value_relu(value* inp1){
	value* out = value_init((inp1->data > 0) ? inp1->data : 0);
	out->op = RELU;
	out->previous[0] = inp1;
	out->previous[0]->ref_count[0]++;
	return out;
}

bool value_equality(value* inp1, value* inp2){
	return (inp1->data == inp2->data) && (inp1->grad == inp2->grad);
}


void grad(value* out){
	switch(out->op){
		case NONE: 
			break; 
		case ADD: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += out->grad; 
			break; 
		case SUB: 
			out->previous[0]->grad += out->grad; 
			out->previous[1]->grad += -out->grad; 
			break; 
		case MUL: 
			out->previous[0]->grad += out->grad * out->previous[1]->data; 
			out->previous[1]->grad += out->grad * out->previous[0]->data; 
			break; 
		case POW:
			out->previous[0]->grad += out->grad * out->previous[1]->data * pow(out->previous[0]->data, out->previous[1]->data - 1) ;
			out->previous[1]->grad += out->grad * (out->data * log(out->previous[0]->data)) ;
			break;
		case SIN: 
			out->previous[0]->grad += out->grad * cos(out->previous[0]->data);
			break; 
		case COS: 
			out->previous[0]->grad += out->grad * -1 * sin(out->previous[0]->data);
			break; 
		case TANH: 
			out->previous[0]->grad += out->grad * (1 - pow(out->data, 2));
			break; 
		case SIGMOID: 
			out->previous[0]->grad += out->grad * out->data * (1 - out->data);
			break; 
		case RELU: 
			out->previous[0]->grad += out->grad * ((out->data > 0 ) ? 1 : 0);
			break; 
		case LOG: 
			out->previous[0]->grad += out->grad * (1.0 / out->data);
			break; 
		case EXP: 
			out->previous[0]->grad += out->grad * (out->data);
			break; 
	}
}


void graph_sort(value* out, value** sorted, int* sorted_size, value** visited, int* visited_size){
	for(int i = 0; i < *visited_size; i++){
		if(visited[i] == out)
			return;
	}
	visited[*visited_size] = out;
	(*visited_size)++;
	for(int i = 0; i < NUM_PREVS; i++){
		if(!out->previous[i])
			break;
		graph_sort(out->previous[i], sorted, sorted_size, visited, visited_size);
	}
	sorted[*sorted_size] = out;
	(*sorted_size)++;
}


void value_backward(value* out){
	size_t graph_size = GRAPH_SIZE;
	value* sorted[graph_size];
	value* visited[graph_size];
	int sorted_size = 0;
	int visited_size = 0;
	graph_sort(out, sorted, &sorted_size, visited, &visited_size);
	out->grad = 1.0;
	for(int i = sorted_size - 1; i >=0; i--){
		grad(sorted[i]);
		value_print(sorted[i]);
		printf("\n");
	}
}









