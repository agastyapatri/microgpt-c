#include "ad.h"
int main(){
	srand(100);
	ad_matrix* x = ad_matrix_random_normal(5, 5, 0, 1);
	ad_matrix* y = ad_matrix_rmsnorm(x);


	ad_matrix_free(x);
	ad_matrix_free(y);

}
