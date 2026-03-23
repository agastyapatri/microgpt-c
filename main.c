#include "ad.h"
#include "utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h> 
#define FILENAME "input.txt"
#define N_EMBD 16							//	embedding dimension 
#define N_HEAD 4							//	number of attention heads 
#define N_LAYER 1							//	number of layers 
#define GPT_BLOCK_SIZE 16					//	maximum sequence length
#define HEAD_DIM (int)(N_EMBD / N_HEAD) 	// dimensionality of each head
#define LEARNING_RATE (float)1.5e-2
#define BETA1 0.85 
#define BETA2 0.99
#define NUM_STEPS 500
#define EPS_ADAM (float)1e-8
#define TEMPERATURE 0.5



int main(void){
	srand(100);
	char documents[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, documents);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, documents);
	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, GPT_BLOCK_SIZE, t.vocab_size);
	parameters*  p = parameters_init(sd);
	gpt_train(sd, p, &t, documents, NUM_STEPS, LEARNING_RATE, GPT_BLOCK_SIZE, BETA1, BETA2, EPS_ADAM);
	gpt_inference(sd, TEMPERATURE, &t, GPT_BLOCK_SIZE);


}







