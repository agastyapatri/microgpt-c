#include "ad.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h> 
#define FILENAME "input.txt"

int main(void){
	srand(42);
	char name_list[NUM_INPUTS][NAMEBUF];
	load_names(FILENAME, name_list);
	tokenizer* t = tokenizer_init(name_list);
	printf("%d\n", t->vocab_size);








	tokenizer_free(t);
	return 0;
}

