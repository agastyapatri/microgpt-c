#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h> 
#include "ad.h"
#define NAMEBUF 128
#define NUM_INPUTS 32000


//	extracts names from the file and shuffles them
void load_names(char* file_name, char word_list[][NAMEBUF]);

//	tokenzier for the name documents;
//		extracts the unique characters from the set of names.
//		each word becomes a list of integers beginning and ending with a BOS token
typedef struct tokenizer {
	int vocab_size;
	int BOS; 
	char uchars[32];
}tokenizer;
void tokenizer_init(tokenizer* t, const char word_list[][NAMEBUF]);
void tokenizer_apply(tokenizer* t, const char* word);



#endif
