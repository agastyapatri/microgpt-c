#include "utils.h"
#include <stdbool.h>
#include <stdio.h> 
#include <string.h> 

void load_names(char* file_name, char name_list[][NAMEBUF]){
	FILE* names = fopen(file_name, "r");
	if(!names){
		fprintf(stderr, "ERROR read_names() could not open %s\n", file_name);
		return;
	}
	char current_name[NAMEBUF];
	int i = 0; 
	while(i < NUM_INPUTS && fgets(current_name, NAMEBUF, names)){
		current_name[strcspn(current_name, "\n")] = '\0';
		strcpy(name_list[i], current_name);
		i++;
	}
	fclose(names);
	for(int i = NUM_INPUTS - 1; i > 0; i--){
		int j = rand() % (i + 1);
		char* temp = name_list[i];
		strncpy(name_list[i], name_list[j], NAMEBUF);
		strncpy(name_list[j], temp, NAMEBUF);
	}
}

tokenizer* tokenizer_init(const char name_list[][NAMEBUF]){
	//	find all the unique characters in the set of names 
	//	assign an integer value to all of them 
	//	BOS = max(token_id) + 1 
	//	vocab_size = BOS + 1
	tokenizer* t = (tokenizer*)malloc(sizeof(tokenizer));
	if(!t)
		return NULL;
	t->vocab_size = 0;
	bool seen[32] = {false};
	for(int i = 0; i < NUM_INPUTS; i++){
		const char* name = name_list[i];
		for(int j = 0; j < NAMEBUF; j++){
			if(!name[j])
				break;
			unsigned char c = name[j];
			if(!seen[c]){
				seen[c] = true;
				t->uchars[t->vocab_size++] = c;
			}
		}
	}
	t->BOS = t->vocab_size + 1;




	return t;
}

void tokenizer_free(tokenizer* t){
	free(t);
}

