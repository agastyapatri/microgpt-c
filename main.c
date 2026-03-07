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



int main(void){
	srand(time(NULL));
	char name_list[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, name_list);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, name_list);


	state_dict* sd = state_dict_init(N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, t.vocab_size);














	return 0;
}

