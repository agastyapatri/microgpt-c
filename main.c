#include "ad.h"
#include "utils.h"
#include <assert.h>
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


void train_gpt(state_dict* sd, parameters* p, tokenizer* t, char docs[][NAMEBUF]);

int main(void){
	srand(100);
	char documents[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, documents);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, documents);
	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, GPT_BLOCK_SIZE, t.vocab_size);
	parameters*  p = parameters_init(sd);


	train_gpt(sd, p, &t, documents);



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
		printf("%d\n", n);


		



	}





}
