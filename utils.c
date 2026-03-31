#include "utils.h"
#include "ad.h"
// #include "ad.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h> 
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include <math.h>

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
	float mu = 0; 
	float sigma = 0.08; 
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
		for(uint j = 0; j < embd_dim*embd_dim; j++){
			sd->attn_wq[i]->data[j]->is_param = true;
			sd->attn_wk[i]->data[j]->is_param = true;
			sd->attn_wv[i]->data[j]->is_param = true;
			sd->attn_wo[i]->data[j]->is_param = true;
		}
		sd->mlp_fc1[i] = ad_matrix_random_normal(4*embd_dim, embd_dim, mu, sigma);
		sd->mlp_fc2[i] = ad_matrix_random_normal(embd_dim, 4*embd_dim, mu, sigma);
		for(uint j = 0; j < 4*embd_dim*embd_dim; j++){
			sd->mlp_fc1[i]->data[j]->is_param = true;
			sd->mlp_fc2[i]->data[j]->is_param = true;
		}

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
	int OFFSET = 0; 
	memcpy(p->param_list + OFFSET, sd->wte->data, sd->wte->nbytes);
	OFFSET += sd->wte->size;
	memcpy(p->param_list + OFFSET, sd->wpe->data, sd->wpe->nbytes);
	OFFSET += sd->wpe->size;
	memcpy(p->param_list + OFFSET, sd->lm_head->data, sd->lm_head->nbytes);
	OFFSET += sd->lm_head->size;
	for(size_t i = 0; i < sd->num_layers; i++){
		memcpy(p->param_list + OFFSET, sd->attn_wq[i]->data, sd->attn_wq[i]->nbytes);
		OFFSET += sd->attn_wq[i]->size;
		memcpy(p->param_list + OFFSET, sd->attn_wk[i]->data, sd->attn_wk[i]->nbytes);
		OFFSET += sd->attn_wk[i]->size;
		memcpy(p->param_list + OFFSET, sd->attn_wv[i]->data, sd->attn_wv[i]->nbytes);
		OFFSET += sd->attn_wv[i]->size;
		memcpy(p->param_list + OFFSET, sd->attn_wo[i]->data, sd->attn_wo[i]->nbytes);
		OFFSET += sd->attn_wo[i]->size;
		memcpy(p->param_list + OFFSET, sd->mlp_fc1[i]->data, sd->mlp_fc1[i]->nbytes);
		OFFSET += sd->mlp_fc1[i]->size;
		memcpy(p->param_list + OFFSET, sd->mlp_fc2[i]->data, sd->mlp_fc2[i]->nbytes);
		OFFSET += sd->mlp_fc2[i]->size;
	}
	return p;
}
void params_free(parameters* p){
	assert(p != NULL);
	free(p->param_list);
	free(p);
}


ad_matrix* gpt_forward(state_dict* sd, int token_id, int pos_id, ad_matrix* keys, ad_matrix* values){
	ad_matrix* x = ad_matrix_alloc(1, sd->wte->cols);
	for(uint i = 0; i < sd->wte->cols; i++){
		ad_value_free(GET(x, 0, i));
		ad_value* tok_emb = GET(sd->wte, token_id, i);
		ad_value* pos_emb = GET(sd->wpe, pos_id, i);
		GET(x, 0, i) = ad_value_add(tok_emb, pos_emb);
	}
	x = ad_matrix_rmsnorm(x);
	ad_matrix* x_residual = x;
	for(uint li = 0; li < sd->num_layers; li++){
		x_residual = x; 
		x = ad_matrix_rmsnorm(x);
		ad_matrix* q = ad_matrix_matmul(x, sd->attn_wq[li]);
		ad_matrix* k = ad_matrix_matmul(x, sd->attn_wk[li]);
		ad_matrix* v = ad_matrix_matmul(x, sd->attn_wv[li]);
		ad_matrix* x_attn = ad_matrix_alloc(1, sd->embd_dim);
		int x_attn_counter = 0;

		//	populating the keys and values
		for(uint j = 0; j < k->cols; j++){
			ad_value_free(keys->data[OFFSET(keys, li, j)]);
			ad_value_free(values->data[OFFSET(values, li, j)]);
			GET(keys, li, j)   = GET(k, 0, j);
			GET(values, li, j) = GET(v, 0, j);
		}
		for(size_t h = 0; h < sd->num_heads; h++){
			int hs = h * sd->head_dim;
			ad_matrix* q_h = ad_matrix_alloc(1, sd->head_dim);
			ad_matrix* k_h = ad_matrix_alloc(sd->num_layers, keys->cols);
			ad_matrix* v_h = ad_matrix_alloc(sd->num_layers, values->cols);
			for(uint _j = 0; _j < sd->head_dim; _j++){
				ad_value_free(q_h->data[OFFSET(q_h, 0, _j)]);
				ad_value_free(k_h->data[OFFSET(k_h, li, _j)]);
				ad_value_free(v_h->data[OFFSET(v_h, li, _j)]);
				GET(q_h, 0, _j)  = GET(q, 0, hs+_j);
				GET(k_h, li, _j) = GET(k, li, hs+_j);
				GET(v_h, li, _j) = GET(v, li, hs+_j);

			}
			ad_matrix* attn_logits = ad_matrix_alloc(1, k_h->rows);
			for(size_t t = 0; t < k_h->rows; t++){
				ad_value* _sum = ad_value_alloc(0.0);
				for(size_t j = 0; j < sd->head_dim; j++){
					ad_value* mul = ad_value_mul(GET(q_h, 0, j), GET(k_h, t, j));
					_sum = ad_value_add(_sum, mul);
				}
				_sum = ad_value_mul(_sum, ad_value_alloc(1.0 / sqrt(sd->head_dim)));
				attn_logits->data[OFFSET(attn_logits, t, 0)] = _sum;
			}
			ad_matrix* attn_weights = ad_matrix_softmax(attn_logits);
			ad_matrix* head_out = ad_matrix_alloc(1, sd->head_dim);
			for(size_t j = 0; j < sd->head_dim; j++){
				ad_value* _sum = ad_value_alloc(0.0);
				for(size_t t = 0; t < v_h->rows; t++){
					ad_value* temp = ad_value_mul(GET(attn_weights, 0, t), GET(v_h, t, j));
					_sum = ad_value_add(_sum, temp);
				}
				GET(head_out, 0, j) = _sum;
			}
			for(uint i = 0; i < head_out->size; i++){
				ad_value_free(GET(x_attn, 0, i + x_attn_counter));
				GET(x_attn, 0, i + x_attn_counter) = GET(head_out, 0, i);
			}
			x_attn_counter+=head_out->size;
		}

		//	output projection + residual
		x = ad_matrix_matmul(x_attn, sd->attn_wo[li]);
		for(uint i = 0; i < x->cols; i++){
			GET(x, 0, i) = ad_value_add(GET(x, 0, i), GET(x_residual, 0, i));
		}

		//	mlp block
		x_residual = x;
		x = ad_matrix_rmsnorm(x);
		x = ad_matrix_matmul(x, sd->mlp_fc1[li]);
		x = ad_matrix_relu(x);
		x = ad_matrix_matmul(x, sd->mlp_fc2[li]);
		for(uint i = 0; i < x->cols; i++){
			GET(x, 0, i) = ad_value_add(GET(x, 0, i), GET(x_residual, 0, i));
		}
	} 
	ad_matrix* logits = ad_matrix_matmul(x, sd->lm_head);
	return logits;

}


void gpt_train(state_dict* sd, parameters* p, tokenizer* t, char docs[][NAMEBUF], int num_steps, float learning_rate, int block_size, float beta1, float beta2, float eps_adam){
	float* m = calloc(p->num_params, sizeof(float));
	memset(m, 0, sizeof(float)*p->num_params);
	float* v = calloc(p->num_params, sizeof(float));
	memset(v, 0, sizeof(float)*p->num_params);
	for(int step = 0; step < num_steps; step++){
		clock_t start = clock();
		char* doc = docs[step % NUM_INPUTS];	//	obtaining a single document
		int	  tokens[NAMEBUF];
		tokenizer_apply(t, doc, tokens);		//	tokenizing the document
		int n = (block_size < (int)strlen(doc)+1) ? block_size : strlen(doc)+1;
		//	forwarding the token sequence through the model; building  up the computational graph.
		ad_matrix* keys   = ad_matrix_alloc(sd->num_layers, sd->attn_wk[0]->cols);
		ad_matrix* values = ad_matrix_alloc(sd->num_layers, sd->attn_wv[0]->cols);
		ad_matrix* losses = ad_matrix_alloc(1, n);
		for(int pos_id = 0; pos_id < n; pos_id++){
			int token_id  = tokens[pos_id];
			int target_id = tokens[pos_id+1];
			ad_matrix* logits = gpt_forward(sd, token_id, pos_id, keys, values); 
			ad_matrix* probs = ad_matrix_softmax(logits);
			ad_value* loss_t = ad_value_log(probs->data[target_id]);
			loss_t = ad_value_mul(loss_t, ad_value_alloc(-1));
			ad_value_free(losses->data[pos_id]);
			losses->data[pos_id] = loss_t;
		}
		ad_value* loss = ad_value_alloc(0.0);
		for(uint i = 0; i < losses->size; i++){
			loss = ad_value_add(loss, losses->data[i]);
		}
		loss = ad_value_mul(loss, ad_value_alloc(1.0 / n));
		ad_value_backward(loss);

		//	adam optimizer param update
		float lr_t = learning_rate * (1 - (float)step / num_steps);
		for(size_t i = 0; i < p->num_params; i++){
			m[i] = beta1 * m[i] + (1 - beta1) * p->param_list[i]->grad;
			v[i] = beta2 * v[i] + (1 - beta2) * (p->param_list[i]->grad * p->param_list[i]->grad);
			float m_hat = m[i]  / (float)(1 - pow(beta1, step+1));
			float v_hat = v[i]  / (float)(1 - pow(beta2, step+1));

			//	param update + zero grad
			p->param_list[i]->data -= lr_t * m_hat / (eps_adam + sqrt(v_hat));
			p->param_list[i]->grad = 0;
		}
		clock_t end = clock();
		printf("step: %d / %d | loss = %lf\ttime = %10.10f\n", step+1, num_steps, loss->data, (double)(end - start) / CLOCKS_PER_SEC);
		ad_matrix_free(losses);
	}
	free(m);
	free(v);
}



void gpt_inference(state_dict* sd, float temperature, tokenizer* t, int block_size){
	printf("\n\nGPT-2 Inference; generating new names\n");
	for(int sample_idx = 0; sample_idx < 20; sample_idx++){
		char sample[NAMEBUF];
		int sample_len = 0;
		int token_id = t->BOS; 
		ad_matrix* keys = ad_matrix_alloc(sd->num_layers, sd->attn_wk[0]->cols);
		ad_matrix* values= ad_matrix_alloc(sd->num_layers, sd->attn_wv[0]->cols);

		for(int pos_id = 0; pos_id < block_size; pos_id++){
			ad_matrix* logits = gpt_forward(sd, token_id ,pos_id, keys, values);

			for(uint i = 0; i < logits->size; i++)
				GET(logits, 0, i)->data /= temperature;
			// fix this to probability weighted random sampling
			ad_matrix* probs = ad_matrix_softmax(logits);


			float r = (float)rand() / (float)RAND_MAX;
			float cumsum = 0.0;
			token_id = t->BOS;
			for(uint i = 0; i < probs->size; i++){
				cumsum += probs->data[i]->data;
				if(r < cumsum){
					token_id = i;
					break;
				}
			}
			if(token_id == t->BOS){
				break;
			}
			sample[sample_len++] = tokenizer_decode(t, token_id);
			ad_matrix_free(probs);
		}
		sample[sample_len] = '\0';
		printf("sample %d; %s\n", sample_idx+1, sample);
		ad_matrix_free(keys);
		ad_matrix_free(values);
	}
}



