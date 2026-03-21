#include "utils.h"
// #include "ad.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h> 
#include <string.h> 
#include <time.h>

int char_cmp(const void* a, const void* b){
	return (*(unsigned char *)a - *(unsigned char* )b);
}

void load_names(char* file_name, char word_list[][NAMEBUF]){
	FILE* names = fopen(file_name, "r");
	if(!names){
		fprintf(stderr, "ERROR read_names() could not open %s\n", file_name);
		return;
	}
	char current_name[NAMEBUF];
	int i = 0; 
	while(i < NUM_INPUTS && fgets(current_name, NAMEBUF, names)){
		current_name[strcspn(current_name, "\n")] = '\0';
		strcpy(word_list[i], current_name);
		i++;
	}
	fclose(names);
	//	shuffling the names
	for(int i = NUM_INPUTS - 1; i > 0; i--){
		char temp[NAMEBUF];
		int j = rand() % (i + 1);
		strncpy(temp, word_list[i], NAMEBUF);
		strncpy(word_list[i], word_list[j], NAMEBUF);
		strncpy(word_list[j], temp, NAMEBUF);
	}
}

void tokenizer_init(tokenizer* t, const char name_list[][NAMEBUF]){
	//	count all the unique characters from the set of names
	bool seen[256];
	memset(seen, 0, 256);
	t->vocab_size = 0;
	for(int i = 0; i < NUM_INPUTS; i++){
		const char* name = name_list[i];
		for(int j = 0; j < NAMEBUF; j++){
			if(!name[j]){
				break;
			}
			unsigned char c = (unsigned char)name[j];
			if(!seen[c]){
				t->uchars[t->vocab_size++] = c;
				seen[c] = 1;
			}
		}
	}
	qsort(t->uchars, t->vocab_size, sizeof(char), char_cmp);
	t->BOS = t->vocab_size;
	t->vocab_size++;
}


//	converts a character to its integer encoding
int tokenizer_encode(tokenizer* t, char c){
	for(int i = 0; i < t->vocab_size-1; i++){
		if(c == t->uchars[i]){
			return i;
		}
	}
	return -999;
}

char tokenizer_decode(tokenizer* t, int n){
	return t->uchars[n];
}

void tokenizer_apply (tokenizer* t, char* doc, int* tokens){
	assert(t      != NULL);
	assert(doc    != NULL);
	assert(tokens != NULL);
	for(uint i = 0; i < strlen(doc); i++)
		tokens[i+1] = tokenizer_encode(t, doc[i]);
	tokens[0] = t->BOS; 
	tokens[strlen(doc)+1] = t->BOS;
}


state_dict* state_dict_init(size_t embd_dim, size_t num_heads, size_t num_layers, size_t block_size, size_t vocab_size){
	double mu = 0; 
	double sigma = 0.08; 
	int head_dim = (int)(embd_dim / num_heads);
	state_dict* sd = (state_dict*)malloc(sizeof(state_dict));
	if(!sd){
		return NULL;
	}
	sd->embd_dim = embd_dim;
	sd->num_heads = num_heads;
	sd->num_layers = num_layers;
	sd->block_size = block_size;
	sd->head_dim = head_dim;
	sd->wte = ad_matrix_random_normal(vocab_size, embd_dim, mu, sigma);
	sd->wpe = ad_matrix_random_normal(block_size, embd_dim, mu, sigma);
	sd->lm_head = ad_matrix_random_normal(vocab_size, embd_dim, mu, sigma);
	sd->attn_wq  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->attn_wq){ 
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}
	sd->attn_wk  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->attn_wk){ 
		free(sd->attn_wq);
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}
	sd->attn_wv  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->attn_wv){ 
		free(sd->attn_wk);
		free(sd->attn_wq);
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}
	sd->attn_wo  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->attn_wo){ 
		free(sd->attn_wv);
		free(sd->attn_wk);
		free(sd->attn_wq);
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}
	sd->mlp_fc1  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->mlp_fc1){ 
		free(sd->attn_wo);
		free(sd->attn_wv);
		free(sd->attn_wk);
		free(sd->attn_wq);
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}

	sd->mlp_fc2  = (ad_matrix**)malloc(num_layers * sizeof(ad_matrix*));
	if(!sd->mlp_fc2){ 
		free(sd->mlp_fc1);
		free(sd->attn_wo);
		free(sd->attn_wv);
		free(sd->attn_wk);
		free(sd->attn_wq);
		ad_matrix_free(sd->wte);
		ad_matrix_free(sd->wpe);
		ad_matrix_free(sd->lm_head);
		free(sd);
		return NULL;
	}

	for(size_t i = 0; i < num_layers; i++){
		sd->attn_wq[i] = ad_matrix_random_normal(embd_dim, embd_dim, mu, sigma);
		sd->attn_wk[i] = ad_matrix_random_normal(embd_dim, embd_dim, mu, sigma);
		sd->attn_wv[i] = ad_matrix_random_normal(embd_dim, embd_dim, mu, sigma);
		sd->attn_wo[i] = ad_matrix_random_normal(embd_dim, embd_dim, mu, sigma);
		sd->mlp_fc1[i] = ad_matrix_random_normal(4*embd_dim, embd_dim, mu, sigma);
		sd->mlp_fc2[i] = ad_matrix_random_normal(embd_dim, 4*embd_dim, mu, sigma);
	}
	return sd;
} 


void state_dict_free(state_dict* sd){
	assert(sd != NULL);
	for(size_t i = 0; i < sd->num_layers; i++){
		ad_matrix_free(sd->attn_wq[i]);
		ad_matrix_free(sd->attn_wk[i]);
		ad_matrix_free(sd->attn_wv[i]);
		ad_matrix_free(sd->attn_wo[i]);
		ad_matrix_free(sd->mlp_fc1[i]);
		ad_matrix_free(sd->mlp_fc2[i]);
	}
	free(sd->mlp_fc2);
	free(sd->mlp_fc1);
	free(sd->attn_wo);
	free(sd->attn_wv);
	free(sd->attn_wk);
	free(sd->attn_wq);
	ad_matrix_free(sd->wte);
	ad_matrix_free(sd->wpe);
	ad_matrix_free(sd->lm_head);
	free(sd);
}

ad_matrix* mlp_forward(ad_matrix* x, state_dict* sd, int li){
	ad_matrix* x_residual = x;
	x = ad_matrix_rmsnorm(x);
	x = ad_matrix_matmul(x, sd->mlp_fc1[li]);
	x = ad_matrix_relu(x);
	x = ad_matrix_matmul(x, sd->mlp_fc2[li]);
	x = ad_matrix_add(x, x_residual);
	return x;
}

parameters* parameters_init(state_dict* sd){
	assert(sd != NULL);
	parameters* p = (parameters*)malloc(sizeof(parameters));
	if(!p)	return NULL;
	p->num_params = 0; 
	p->num_params += sd->wte->size;
	p->num_params += sd->wpe->size;
	p->num_params += sd->lm_head->size; 
	for(size_t i = 0; i < sd->num_layers; i++){
		p->num_params += sd->attn_wq[i]->size;
		p->num_params += sd->attn_wk[i]->size;
		p->num_params += sd->attn_wv[i]->size;
		p->num_params += sd->attn_wo[i]->size;
		p->num_params += sd->mlp_fc1[i]->size;
		p->num_params += sd->mlp_fc2[i]->size;
	}
	p->param_list = (ad_value**)malloc(p->num_params * sizeof(ad_value*));
	if(!p->param_list){
		free(p);
		return NULL;
	}
	int offset = 0; 
	memcpy(p->param_list + offset, sd->wte->data, sd->wte->nbytes);
	offset += sd->wte->size;
	memcpy(p->param_list + offset, sd->wpe->data, sd->wpe->nbytes);
	offset += sd->wpe->size;
	memcpy(p->param_list + offset, sd->lm_head->data, sd->lm_head->nbytes);
	offset += sd->lm_head->size;
	for(size_t i = 0; i < sd->num_layers; i++){
		memcpy(p->param_list + offset, sd->attn_wq[i]->data, sd->attn_wq[i]->nbytes);
		offset += sd->attn_wq[i]->size;
		memcpy(p->param_list + offset, sd->attn_wk[i]->data, sd->attn_wk[i]->nbytes);
		offset += sd->attn_wk[i]->size;
		memcpy(p->param_list + offset, sd->attn_wv[i]->data, sd->attn_wv[i]->nbytes);
		offset += sd->attn_wv[i]->size;
		memcpy(p->param_list + offset, sd->attn_wo[i]->data, sd->attn_wo[i]->nbytes);
		offset += sd->attn_wo[i]->size;
		memcpy(p->param_list + offset, sd->mlp_fc1[i]->data, sd->mlp_fc1[i]->nbytes);
		offset += sd->mlp_fc1[i]->size;
		memcpy(p->param_list + offset, sd->mlp_fc2[i]->data, sd->mlp_fc2[i]->nbytes);
		offset += sd->mlp_fc2[i]->size;
	}
	return p;
}
void params_free(parameters* p){
	assert(p != NULL);
	free(p->param_list);
	free(p);
}



ad_matrix* gpt(state_dict* sd, int token_id, int pos_id, ad_matrix* keys, ad_matrix* values){
	ad_matrix* x = ad_matrix_alloc(1, sd->wte->cols);
	for(uint i = 0; i < sd->wte->cols; i++){
		ad_value_free(x->data[offset(x, 0, i)]);
		ad_value* tok_emb = sd->wte->data[offset(sd->wte, 0, i)]; 
		ad_value* pos_emb = sd->wpe->data[offset(sd->wpe, 0, i)]; 
		x->data[offset(x, 0, i)] = ad_value_add(tok_emb, pos_emb);
	}
	x = ad_matrix_rmsnorm(x);
	ad_matrix* x_residual = x;
	for(uint li = 0; li < sd->num_layers; li++){
		x_residual = x; 
		x = ad_matrix_rmsnorm(x);
		ad_matrix* q = ad_matrix_matmul(x, sd->attn_wq[li]);
		ad_matrix* k = ad_matrix_matmul(x, sd->attn_wk[li]);
		ad_matrix* v = ad_matrix_matmul(x, sd->attn_wv[li]);
		for(uint j = 0; j < k->cols; j++){
			ad_value_free(keys->data[offset(keys, li, j)]);
			ad_value_free(values->data[offset(values, li, j)]);
			keys->data[offset(keys, li, j)] = k->data[offset(k, 0, j)];
			values->data[offset(values, li, j)] = v->data[offset(v, 0, j)];
		}

	}

}



