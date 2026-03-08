#include "ad.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h> 
#define FILENAME "input.txt"
#define N_EMBD 16						//	embedding dimension 
#define N_HEAD 4						//	number of attention heads 
#define N_LAYER 1						//	number of layers 
#define BLOCK_SIZE 16					//	maximum sequence length
#define HEAD_DIM (int)(N_EMBD / N_HEAD) // dimensionality of each head
#define LEARNING_RATE (double)1e-2
#define BETA1 0.85 
#define BETA2 0.99
#define NUM_STEPS 1000 
#define EPS_ADAM (double)1e-8





int main(void){
	srand(time(NULL));
	char name_list[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, name_list);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, name_list);
	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, t.vocab_size);
	params* params = params_init(sd);


	for(int i = 0; i < NUM_INPUTS; i++){
		const char* name = name_list[i];
		printf("%s\t", name );
		tokenizer_apply(&t, name);
		printf("\n");
		if(i == 5)
			break;
	}


	









	state_dict_free(sd);
	params_free(params);
	return 0;
}

