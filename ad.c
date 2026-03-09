#include "ad.h"
#include <assert.h>
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
		case(DIV):
			return "div";
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

ad_value* ad_value_alloc(double data){
	ad_value* a = (ad_value*)malloc(sizeof(ad_value));
	if(!a)	return NULL;
	a->data = data; 
	a->grad = 0;
	a->op = NONE;
	a->previous[0] = NULL;
	a->previous[1] = NULL;
	a->ref_count = 1;
	return a;
}

double rand_normal(double mu, double sigma){
	double n2 = 0.0; 
	double n2_cached = 0.0; 
	if(!n2_cached){
		double u1 = rand_double();
		double u2 = rand_double();
		double r = sqrt(-2.0 * log(u1));
		double theta = 2 * PI * u2;
		n2 = r * sin(theta);
		n2_cached = 1;
		return (r * cos(theta) * sigma + mu);
	}
	else{
		n2_cached = 0;
		return (n2*sigma + mu);
	}
}


ad_value* ad_value_rand_normal(double mu, double sigma){
	return ad_value_alloc(rand_normal(mu, sigma));
}

ad_value* ad_value_random_gauss(double mu, double sigma){
	double n2 = 0.0; 
	double n2_cached = 0.0; 
	double data;
	if(!n2_cached){
		// double u1 = rand_double();
		double u1 = rand() / (double)RAND_MAX;
		double u2 = rand() / (double)RAND_MAX;
		double r = sqrt(-2.0 * log(u1));
		double theta = 2 * PI * u2;
		n2 = r * sin(theta);
		n2_cached = 1;
		data = r * cos(theta) * sigma + mu;
	}
	else{
		n2_cached = 0;
		data = n2*sigma + mu;
	}
	ad_value* out = ad_value_alloc(data);
	return out;
}

void ad_value_free(ad_value* val){
	val->ref_count--;
	if(val->ref_count == 0){
		if(val->previous[0])
			ad_value_free(val->previous[0]);
		if(val->previous[1])
			ad_value_free(val->previous[1]);
		free(val);
	}
}


void ad_value_print(ad_value* val){
	printf("ad_value(data: %lf, grad: %lf, op: %s)", val->data, val->grad, get_optype_string(val->op));
}

ad_value* ad_value_add(ad_value* inp1, ad_value* inp2){
	ad_value* out = ad_value_alloc(inp1->data + inp2->data);
	out->op = ADD;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count++;
	out->previous[1]->ref_count++;

	return out;
}

ad_value* ad_value_sub(ad_value* inp1, ad_value* inp2){
	ad_value* out = ad_value_alloc(inp1->data - inp2->data);
	out->op = SUB;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count++;
	out->previous[1]->ref_count++;
	return out;
}
ad_value* ad_value_mul(ad_value* inp1, ad_value* inp2){
	ad_value* out = ad_value_alloc(inp1->data * inp2->data);
	out->op = MUL;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count++;
	out->previous[1]->ref_count++;
	return out;
}

ad_value* ad_value_div	(ad_value* inp1, ad_value* inp2){
	ad_value* out = ad_value_alloc(inp1->data / inp2->data);
	out->op = DIV;
	out->previous[0] = inp1;
	out->previous[1] = inp2;
	out->previous[0]->ref_count++;
	out->previous[1]->ref_count++;
	return out;

}



ad_value* ad_value_pow(ad_value* inp1, ad_value* exponent){
	ad_value* out = ad_value_alloc(pow(inp1->data, exponent->data));
	out->op = POW;
	out->previous[0] = inp1;
	out->previous[1] = exponent;
	out->previous[0]->ref_count++;
	out->previous[1]->ref_count++;
	return out;
}

ad_value* ad_value_sigmoid(ad_value* inp1){
	ad_value* out = ad_value_alloc(1.0 / (1 + exp(-(inp1->data))));
	out->op = SIGMOID;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}

ad_value* ad_value_tanh(ad_value* inp1){
	ad_value* out = ad_value_alloc(tanh(inp1->data));
	out->op = TANH;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}
ad_value* ad_value_log(ad_value* inp1){
	ad_value* out = ad_value_alloc(log(inp1->data));
	out->op = LOG;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}
ad_value* ad_value_exp(ad_value* inp1){
	ad_value* out = ad_value_alloc(exp(inp1->data));
	out->op = EXP;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}

ad_value* ad_value_sin(ad_value* inp1){
	ad_value* out = ad_value_alloc(sin(inp1->data));
	out->op = SIN;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}

ad_value* ad_value_cos(ad_value* inp1){
	ad_value* out = ad_value_alloc(cos(inp1->data));
	out->op = COS;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}


ad_value* ad_value_relu(ad_value* inp1){
	ad_value* out = ad_value_alloc((inp1->data > 0) ? inp1->data : 0);
	out->op = RELU;
	out->previous[0] = inp1;
	out->previous[0]->ref_count++;
	return out;
}

bool ad_value_equality(ad_value* inp1, ad_value* inp2){
	return (inp1->data == inp2->data) && (inp1->grad == inp2->grad);
}


void grad(ad_value* out){
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
		case DIV: 
			out->previous[0]->grad += out->grad * 1 / out->previous[1]->data; 
			out->previous[1]->grad += out->grad * (-out->previous[0]->data / (pow(out->previous[1]->data, 2))); 
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


void graph_sort(ad_value* out, ad_value** sorted, int* sorted_size, ad_value** visited, int* visited_size){
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


void ad_value_backward(ad_value* out){
	size_t graph_size = GRAPH_SIZE;
	ad_value* sorted[graph_size];
	ad_value* visited[graph_size];
	int sorted_size = 0;
	int visited_size = 0;
	graph_sort(out, sorted, &sorted_size, visited, &visited_size);
	out->grad = 1.0;
	for(int i = sorted_size - 1; i >=0; i--){
		grad(sorted[i]);
	}
}

ad_matrix* ad_matrix_alloc(uint nrows, uint ncols){
	uint size = nrows * ncols; 
	ad_matrix* m = (ad_matrix*)malloc(sizeof(ad_matrix));
	if(!m){
		return NULL;
	}
	AD_MATRIX_SHAPE_INIT(m, nrows, ncols);
	m->data = (ad_value**)malloc(size * sizeof(ad_value*));
	if(!m->data){
		free(m);
		return NULL;
	}
	for(uint i = 0; i < size; i++){
		m->data[i] = ad_value_alloc(0);
	} 
	return m;
}

void ad_matrix_print(ad_matrix* m){
	assert(m!=NULL);
	printf("ad_matrix[[");
	for(uint i = 0; i < m->rows; i++){
		if(i > 0)
			printf("          [");
		for(uint j = 0; j < m->cols; j++){
			double data = m->data[offset(m, i, j)]->data;
			if(data >= 0)
				printf("%10.8f", data);
			else
				printf("%10.7f", data);
			if(j < m->cols - 1)
				printf("  ");
		} 
		printf("]");
		if(i < m->rows - 1)
			printf("\n");
	}
	printf("]\n");
}


ad_matrix* ad_matrix_random_normal(int nrows, int ncols, double mu, double sigma){
	ad_matrix* m = ad_matrix_alloc(nrows, ncols);
	for(uint i = 0; i < m->size; i++)
		m->data[i]->data = rand_normal(mu, sigma);
	return m;
}

void ad_matrix_free(ad_matrix* m){
	assert(m!=NULL);
	for(uint i = 0; i < m->size; i++){
		ad_value_free(m->data[i]);
	}
	free(m->data);
	free(m);
}

double ad_matrix_sum(ad_matrix* m){
	assert(m!=NULL);
	double _sum = 0.0; 
	for(uint i = 0; i < m->size; i++)
		_sum += m->data[i]->data;
	return _sum;
}

double ad_matrix_mean(ad_matrix* m){
	assert(m!=NULL);
	return ad_matrix_sum(m)/m->size;
}

double ad_matrix_std(ad_matrix* m){
	assert(m!=NULL);
	double mu = ad_matrix_mean(m);
	double sigma = 0.0;
	for(uint i = 0; i < m->size; i++){
		sigma += (m->data[i]->data - mu)*(m->data[i]->data - mu);
	}
	sigma/=m->size;
	return sqrt(sigma);
}

double 	ad_matrix_max(ad_matrix* m){
	assert(m!=NULL);
	double max = 0;
	for(uint i = 0; i < m->size; i++){
		if(m->data[i]->data > max){
			max = m->data[i]->data;
		}
	}
	return max;
}

double 	ad_matrix_min(ad_matrix* m){
	assert(m!=NULL);
	double max = (double)1e9;
	for(uint i = 0; i < m->size; i++){
		if(m->data[i]->data < max){
			max = m->data[i]->data;
		}
	}
	return max;
}

ad_matrix* ad_matrix_softmax(ad_matrix* x){
	assert(x!=NULL);
	ad_matrix* out = ad_matrix_alloc(x->rows, x->cols);
	for(uint i = 0; i < x->rows; i++){
		ad_value** exps = malloc(x->cols * sizeof(ad_value*));
		for(uint j = 0; j < x->cols; j++){
			exps[j] = ad_value_exp(x->data[offset(x, i, j)]);
		}
		ad_value* sum = ad_value_alloc(0);
		for(uint j = 0; j < x->cols; j++){
			ad_value* new_sum = ad_value_add(sum, exps[j]);
			ad_value_free(sum);
			sum = new_sum;
		}
		for(uint j = 0; j < x->cols; j++){
			ad_value_free(out->data[offset(out, i, j)]);
			out->data[offset(out, i, j)] = ad_value_div(exps[j], sum); // assign the value of the division to the output
			ad_value_free(exps[j]);
		}
		ad_value_free(sum);
		free(exps);
	}
	return out;
}

ad_matrix* ad_matrix_rmsnorm(ad_matrix* x){
	assert(x != NULL);
	ad_matrix* out = ad_matrix_alloc(x->rows, x->cols);
	ad_value* rms = ad_value_alloc(0);
	ad_value* constant = ad_value_alloc(2);
	for(uint i = 0; i < x->rows; i++){
		for(uint j = 0; j < x->cols; j++){
			ad_value* exponent = ad_value_pow(x->data[offset(x, i, j)], constant);
			ad_value* new_sum = ad_value_add(rms, exponent);
			ad_value_free(rms);
			ad_value_free(exponent);
			rms = new_sum;
		}
	}
	ad_value_free(constant);
	ad_value* scale = ad_value_alloc(1.0/x->size);
	ad_value* scaled = ad_value_mul(rms, scale);
	constant = ad_value_alloc(0.5);
	ad_value* rms_final = ad_value_pow(scaled, constant);
	ad_value_free(constant);
	ad_value_free(scale);
	ad_value_free(scaled);
	ad_value_free(rms);

	for(uint i = 0; i < x->size; i++){
		ad_value_free(out->data[i]);
		out->data[i] = ad_value_div(x->data[i], rms_final);
	}
	ad_value_free(rms_final);
	return out;
}


ad_matrix* ad_matrix_matmul(ad_matrix* x, ad_matrix* y){
	assert(x!=NULL);
	assert(y!=NULL);
	assert(x->cols == y->rows);
	ad_matrix* out = ad_matrix_alloc(x->rows, y->cols);
	for(uint i = 0; i < x->rows; i++){
		for(uint k = 0; k < x->cols; k++){
			double x_ik = get(x, i, k);
			for(uint j = 0; j < y->cols; j++){
				out->data[offset(out, i, j)]->data += x_ik * y->data[offset(y, k, j)]->data;
			}
		}
	}
	return out;
}

ad_matrix* ad_matrix_get_row(ad_matrix* m, int row_idx){
	assert(m!=NULL);
	ad_matrix* row = ad_matrix_alloc(1, m->cols);
	for(uint j = 0; j < m->cols; j++){
		set(row, 0, j, get(m, row_idx, j));
	}
	return row;
}

