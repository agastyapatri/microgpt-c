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
	
	ad_matrix* x = ad_matrix_alloc(1, 4);
	x->data[0]->data = 1;
	x->data[1]->data = 2;
	x->data[2]->data = 3;
	x->data[3]->data = 4;
	ad_matrix* out = ad_matrix_softmax(x);
	int target = 3;
	ad_value* loss = ad_value_neg(ad_value_log(out->data[target]));
	ad_value_backward(loss);
	ad_value_print(loss);
	printf("\n\n");

	for(uint i = 0; i < x->size; i++){
		ad_value_print(x->data[i]);
		printf("\n");
	}



	

	



}

	// printf("Autograd Gradients: \n");
	// for(uint i = 0 ;i < out->size; i++)
	// 	printf("%lf ", x->data[i]->grad);
	// printf("\n");
	//
	// printf("Numerical Gradients: \n");
	//
	// double h = (double)1e-5;
	// for(uint i = 0 ;i < out->size; i++){
	// 	x->data[i]->data += h;
	// 	double sum = 0.0;
	// 	ad_matrix* out2 = ad_matrix_rmsnorm(x);
	// 	for(uint j = 0; j < out2->size; j++){
	// 		sum += out2->data[i]->data;
	// 	}
	//
	// 	x->data[i]->data -= 2*h;
	// 	double sum2 = 0.0;
	// 	ad_matrix* out3 = ad_matrix_rmsnorm(x);
	// 	for(uint j = 0; j < out2->size; j++){
	// 		sum2 += out3->data[i]->data;
	// 	}
	// 	x->data[i]->data += h;
	// 	printf("%lf ", (sum - sum2) / h);
	// }
