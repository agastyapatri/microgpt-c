#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h> 
#define NAMEBUF 128
#define NUM_INPUTS 32000

//	extracts names from the file and shuffles them
void load_names(char* file_name, char name_list[][NAMEBUF]);


typedef struct tokenizer {
	int vocab_size;
	int BOS; 
	char uchars[32];
}tokenizer;
void tokenizer_init(tokenizer* t, const char name_list[][NAMEBUF]);
void tokenizer_apply(tokenizer* t, const char* word);





#endif
