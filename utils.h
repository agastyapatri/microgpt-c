#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h> 
// #include "ad.h"
#include "matrix.h"
#define NAMEBUF 128
#define NUM_INPUTS 32000


void load_names(char* file_name, char word_list[][NAMEBUF]);

typedef struct tokenizer {
	int vocab_size;
	int BOS; 
	char uchars[32];
}tokenizer;

void tokenizer_init(tokenizer* t, const char word_list[][NAMEBUF]);
matrix* tokenizer_apply(tokenizer* t, const char* word);
int tokenizer_get_token(tokenizer* t, const char c);

typedef struct state_dict{
	size_t 		embd_dim;		//	embedding dimensions 
	size_t 		num_heads;		//	number of attention heads 
	size_t 		num_layers;		//	number of layers 
	size_t 		block_size;		//	maximum sequence length
	size_t 		head_dim;		//	dimension of each head
	matrix*  wte; 
	matrix*  wpe; 
	matrix*  lm_head; 		//	
	matrix** attn_wq;		//	array of query weight matrices for each layer
	matrix** attn_wk;		// 	array of key weight matrices for each layer	
	matrix** attn_wv;   		// 	array of value weight matrices for each layer	 
	matrix** attn_wo;		// 	array of output?? matrices for each layer
	matrix** mlp_fc1;		// 	array of fully connected portions of each layer
	matrix** mlp_fc2;		// 	array of fully connected portions of each layer
} state_dict;
state_dict* state_dict_init(size_t embd_dim, size_t num_heads, size_t num_layers, size_t block_size, size_t vocab_size);
void state_dict_free(state_dict* sd);


// typedef struct params {
// 	matrix* param_list; 
// 	size_t num_params;
// } params;
// params* params_init(state_dict* sd);
// void params_free(params* p);



matrix* gpt(state_dict* sd, int token_id, int pos_id, matrix* keys, matrix* values);







#endif
