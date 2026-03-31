#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h> 
#include "ad.h"
#define NAMEBUF 32
#define NUM_INPUTS 32000


void load_names(char* file_name, char word_list[][NAMEBUF]);

typedef struct tokenizer {
	int vocab_size;
	int BOS; 
	char uchars[32];
}tokenizer;

void tokenizer_init	 (tokenizer* t, const char word_list[][NAMEBUF]);
int tokenizer_encode (tokenizer* t, char c);
char tokenizer_decode(tokenizer* t, int n);
void tokenizer_apply (tokenizer* t, char* doc, int* tokens);




typedef struct state_dict{
	size_t 		embd_dim;		//	embedding dimensions 
	size_t 		num_heads;		//	number of attention heads 
	size_t 		num_layers;		//	number of layers 
	size_t 		block_size;		//	maximum sequence length
	size_t 		head_dim;		//	dimension of each head
	ad_matrix*  wte; 			//	weight token embedding; each row is the learned vector representation of one token 
	ad_matrix*  wpe; 			//	tells the model where in the sequence each token is  
	ad_matrix*  lm_head; 		//	
	ad_matrix** attn_wq;		//	array of query weight matrices for each layer
	ad_matrix** attn_wk;		// 	array of key weight matrices for each layer	
	ad_matrix** attn_wv;   		// 	array of value weight matrices for each layer	 
	ad_matrix** attn_wo;		// 	array of output?? matrices for each layer
	ad_matrix** mlp_fc1;		// 	array of fully connected portions of each layer
	ad_matrix** mlp_fc2;		// 	array of fully connected portions of each layer
} state_dict;
state_dict* state_dict_init(size_t embd_dim, size_t num_heads, size_t num_layers, size_t block_size, size_t vocab_size);
void state_dict_free(state_dict* sd);



//	parameters is a flat representation of the gpt-2 state_dict. 
typedef struct params {
	ad_value** param_list; 
	size_t num_params;
} parameters;
parameters* parameters_init(state_dict* sd);
void parameters_free(parameters* p);



ad_matrix* gpt_forward(state_dict* sd, int token_id, int pos_id, ad_matrix* keys, ad_matrix* values);
void 	   gpt_train(state_dict* sd, parameters* p, tokenizer* t, char docs[][NAMEBUF], int num_steps, float learning_rate, int block_size, float beta1, float beta2, float eps_adam);
void 	   gpt_inference(state_dict* sd, float temperature, tokenizer* t, int block_size);






typedef struct {
	int n_pos; 
	int capacity;
	int n_layers; 
	ad_matrix*** keys; 
	ad_matrix*** values; 
} kv_cache;

kv_cache* kv_cache_init(int n_layers, int init_capacity);
void kv_cache_append(kv_cache* kv, int layer, ad_matrix* k, ad_matrix* v);
void kv_cache_free(kv_cache* kv);
ad_matrix* gpt_forward_alt(state_dict* sd, int token_id, int pos_id, kv_cache* kv);
void 	   gpt_train_alt(state_dict* sd, parameters* p, tokenizer* t, char docs[][NAMEBUF], int num_steps, float learning_rate, int block_size, float beta1, float beta2, float eps_adam);




#endif
