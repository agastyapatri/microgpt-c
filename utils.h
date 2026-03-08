#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h> 
#include "ad.h"
#define NAMEBUF 128
#define NUM_INPUTS 32000


void load_names(char* file_name, char word_list[][NAMEBUF]);

typedef struct tokenizer {
	int vocab_size;
	int BOS; 
	char uchars[32];
}tokenizer;

void tokenizer_init(tokenizer* t, const char word_list[][NAMEBUF]);
void tokenizer_apply(tokenizer* t, const char* word);


typedef struct state_dict{
	uint embd_dim;			//	embedding dimensions 
	uint num_heads;			//	number of attention heads 
	uint num_layers;		//	number of layers 
	uint block_size;		//	maximum sequence length
	uint head_dim;			//	dimension of each head
	ad_matrix* wte; 
	ad_matrix* wpe; 
	ad_matrix* lm_head; 	//	
	ad_matrix** attn_wq;	//	array of query weight matrices for each layer
	ad_matrix** attn_wk;	// 	array of key weight matrices for each layer	
	ad_matrix** attn_wv;   	// 	array of value weight matrices for each layer	 
	ad_matrix** attn_wo;	// 	array of output?? matrices for each layer
	ad_matrix** mlp_fc1;	// 	array of fully connected portions of each layer
	ad_matrix** mlp_fc2;	// 	array of fully connected portions of each layer
} state_dict;


state_dict* state_dict_init(uint embd_dim, uint num_heads, uint num_layers, uint block_size, uint vocab_size);
void state_dict_free(state_dict* sd);






#endif
