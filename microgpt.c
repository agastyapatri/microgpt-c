#include "autograd.h"
#include "matrix.h"
#include "utils.h"
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
	char name_list[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, name_list);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, name_list);
	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, GPT_BLOCK_SIZE, t.vocab_size);

















	

	//	TRAINING 
	// for(int e = 0; e < NUM_STEPS; e++){
	// 	const char* doc = name_list[e % NUM_INPUTS];
	// 	ad_matrix* tokens = tokenizer_apply(&t, doc);
	// 	int n = (BLOCK_SIZE < tokens->size-1) ? BLOCK_SIZE : tokens->size-1;
	// 	ad_matrix* keys = ad_matrix_alloc(N_LAYER, N_LAYER);
	// 	ad_matrix* values = ad_matrix_alloc(N_LAYER, N_LAYER);
	// 	ad_matrix* losses = ad_matrix_alloc(1, n);
	// 	for(int pos_id = 0; pos_id < n; pos_id++){
	// 		int token_id = get(tokens, 0, pos_id);
	// 		int target_id = get(tokens, 0, pos_id+1);
	// 		ad_matrix* logits =  gpt(sd, token_id, pos_id, keys, values);
	//
	//
	//
	//
	// 		ad_matrix_free(logits);
	// 		break;
	// 	}
	// 	printf("\n");
	// 	break;
	//
	// 	ad_matrix_free(tokens);
	// 	ad_matrix_free(keys);
	// 	ad_matrix_free(values);
	// 	ad_matrix_free(losses);
	// }
	//
	//
	// state_dict_free(sd);
	// params_free(params);
	// return 0;
}


