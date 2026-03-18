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


int main(void){
	srand(100);
	char name_list[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, name_list);
	tokenizer t = {.BOS = 0, .vocab_size = 0};
	tokenizer_init(&t, name_list);
	
	// ad_matrix* x = ad_matrix_alloc(1, 5);
	// for(uint i = 0; i < x->size; i++)
	// 	x->data[i]->data = i+1;
	// ad_matrix* y = ad_matrix_rmsnorm(x);
	// for(uint i = 0; i < x->size; i++)
	// 	printf("%lf ", x->data[i]->grad);

	// x = [1.0, 2.0, 3.0, 4.0]
	// ms = (1 + 4 + 9 + 16) / 4 = 7.5
	// scale = 1 / sqrt(7.5 + 1e-5) = 0.3651
	// expected output = [0.3651, 0.7303, 1.0954, 1.4606]
	

	



}

