#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <nmmintrin.h>
#include <immintrin.h>

int a[16] = { 1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6};
int b[16] = { 2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7};
int c[16];
int n = 4;

int normal_func() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i*n + j] = 0;
			for (int k = 0; k < n; k++) {
				c[i*n + j] += a[i*n + k] * b[k*n + j];
			}
		}
	}
	return 1;
}

int normal_func_Qpar() {
#pragma loop(hint_parallel(6))
#pragma loop(ivdep)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i*n + j] = 0;
			for (int k = 0; k < n; k++) {
				c[i*n + j] += a[i*n + k] * b[k*n + j];
			}
		}
	}
	return 1;
}

int main() {
	int start_clock = (float)clock();
	
	
	//Matrix multiplication
#pragma loop(no_parallel)
	for (int ind = 0; ind < 1000000; ind++)
	{	
		normal_func_Qpar();
		__m256i m0;
	}

	printf("Elaspse time=%d.\n", (int)clock() - start_clock);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d ",c[i*n+j]);
		}
		printf("\n");
	}

	char temp_chr='0';
	scanf("%c", &temp_chr);
	return 0;
}