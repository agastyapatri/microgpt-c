#include "ad.h"
#include <stdio.h>
#include <string.h>
#define FILENAME "input.txt"
#define NAMEBUF 128
#define NUM_INPUTS 32000

//	returns a list of names read from the file; assumed at the largest name is 63 characters
// void read_names(char* file_name, char* name_list[NUM_INPUTS]);
void read_names(char* file_name, char name_list[][NAMEBUF]);



int main(void){
	char name_list[NUM_INPUTS][NAMEBUF];
	read_names(FILENAME, name_list);


}

void read_names(char* file_name, char name_list[][NAMEBUF]){
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
}
