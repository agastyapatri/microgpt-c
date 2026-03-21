#include "ad.h"
#include "utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h> 
#define FILENAME "input.txt"
#define N_EMBD 16						//	embedding dimension 
#define N_HEAD 4						//	number of attention heads 
#define N_LAYER 1						//	number of layers 
#define GPT_BLOCK_SIZE 16					//	maximum sequence length
#define HEAD_DIM (int)(N_EMBD / N_HEAD) // dimensionality of each head
#define LEARNING_RATE (double)1e-2
#define BETA1 0.85 
#define BETA2 0.99
#define NUM_STEPS 1000 
#define EPS_ADAM (double)1e-8



int main(void){
	srand(100);
	char documents[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, documents);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, documents);
	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, GPT_BLOCK_SIZE, t.vocab_size);
	parameters*  p = parameters_init(sd);
	// train_gpt(sd, p, &t, documents);
	
	char* doc = documents[0];
	int	  tokens[NAMEBUF];
	tokenizer_apply(&t, doc, tokens);		//	tokenizing the document
	int n = (GPT_BLOCK_SIZE < strlen(doc)+1) ? GPT_BLOCK_SIZE : strlen(doc)+1;
	int tok_len = strlen(doc) + 2;

	ad_matrix* keys   = ad_matrix_alloc(sd->num_layers, sd->attn_wk[0]->cols);
	ad_matrix* values = ad_matrix_alloc(sd->num_layers, sd->attn_wv[0]->cols);
	ad_matrix* losses = ad_matrix_alloc(sd->num_layers, n);
	for(int pos_id = 0; pos_id < n; pos_id++){
		int token_id  = tokens[pos_id];
		int target_id = tokens[pos_id+1];
		ad_matrix* logits = gpt(sd, token_id, pos_id, keys, values); 
		// ad_matrix* probs = ad_matrix_softmax(logits);
		// ad_value* loss_t = ad_value_log(probs->data[target_id]);
		// loss_t = ad_value_mul(loss_t, ad_value_alloc(-1));
		// losses->data[pos_id] = loss_t;
		break;
	}


}

void train_gpt(state_dict* sd, parameters* p, tokenizer* t, char docs[][NAMEBUF]){
	double m[p->num_params];					//	Adam: first moment buffer
	memset(m, 0, sizeof(double)*p->num_params);
	double v[p->num_params];					//	Adam: second moment buffer 
	memset(v, 0, sizeof(double)*p->num_params);
	for(int step = 0; step < NUM_STEPS; step++){
		char* doc = docs[step % NUM_INPUTS];	//	obtaining a single document
		int	  tokens[NAMEBUF];
		tokenizer_apply(t, doc, tokens);		//	tokenizing the document
		int n = (GPT_BLOCK_SIZE < strlen(doc)+1) ? GPT_BLOCK_SIZE : strlen(doc)+1;
		int tok_len = strlen(doc) + 2;

		//	forwarding the token sequence through the model; building  up the computational graph.
		ad_matrix* keys   = ad_matrix_alloc(sd->num_layers, sd->attn_wk[0]->cols);
		ad_matrix* values = ad_matrix_alloc(sd->num_layers, sd->attn_wv[0]->cols);
		ad_matrix* losses = ad_matrix_alloc(sd->num_layers, n);
		for(int pos_id = 0; pos_id < n; pos_id++){
			int token_id  = tokens[pos_id];
			int target_id = tokens[pos_id+1];
			ad_matrix* logits = gpt(sd, token_id, pos_id, keys, values); 
			ad_matrix* probs = ad_matrix_softmax(logits);
			ad_value* loss_t = ad_value_log(probs->data[target_id]);
			loss_t = ad_value_mul(loss_t, ad_value_alloc(-1));
			losses->data[pos_id] = loss_t;
		}
		ad_value* loss = ad_value_alloc(0.0);
		for(uint i = 0; i < losses->size; i++){
			loss = ad_value_add(loss, losses->data[i]);
		}
		loss = ad_value_mul(loss, ad_value_alloc(1.0 / n));
		ad_value_backward(loss);

		//	adam optimizer param update
		double lr_t = LEARNING_RATE * (1 - (double)step / NUM_STEPS);
		for(size_t i = 0; i < p->num_params; i++){
			m[i] = BETA1 * m[i] + (1 - BETA1) * p->param_list[i]->grad;
			v[i] = BETA2 * v[i] + (1 - BETA2) * (p->param_list[i]->grad * p->param_list[i]->grad);
			double m_hat = m[i]  / (double)(1 - pow(BETA1, step+1));
			double v_hat = v[i]  / (double)(1 - pow(BETA2, step+1));
			p->param_list[i]->data -= lr_t * m_hat / (EPS_ADAM + sqrt(v_hat));
			p->param_list[i]->grad = 0;
		}
		printf("step: %d / %d | loss = %lf\n", step+1, NUM_STEPS, loss->data);
	}
}
