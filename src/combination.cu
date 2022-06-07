
#include <stdio.h>

#include "combination.hpp"

__constant__ double factorial[65];
double factorial_h[65];


__device__ __host__ void print_mask(uint64_t mask, uint32_t size) {

    char res[100]; 

    uint64_t bit = 1ll << (size - 1);
    int count = 0;

    while(bit > 0) {

        if(mask & bit) {
            res[count] = '1';
        } else{
            res[count] = '0';
        }

        bit = bit >> 1;
        count++;

    }

    res[size] = '\n';
    res[size + 1] = '\0';

    printf("%s", res);


}


__device__ __host__ int next_combination(int32_t n, int32_t k, uint64_t* mask) {

    if(*mask == 0) {

        *mask = (1ll << k) - 1ll;

    } else {

        uint64_t c = (*mask) & -(*mask);
        uint64_t r = (*mask) + c;
        *mask = r | (((r ^ (*mask)) >> 2)/c);

    }

    return *mask <= (1ll << n) - (1ll << (n-k));

}


__device__ uint64_t ith_combination(int32_t n, int32_t r, uint64_t i) {
	
	uint64_t mask = 0llu;

	while(n > 0) {

		uint64_t y = 0;

		if(n > r) {
			y = nCr(n-1, r);
		}

		if(i >= y) {

			i = i - y;
			mask = mask | (1llu << (n-1));
			r = r - 1;

		} else {

			mask = mask & ~(1llu << (n-1));

		}

		n = n-1;

	}

	return mask;

}


__host__ void initialize_factorial_table() {

    factorial_h[0] = 1;

    for(int32_t i=1; i<65; i++) {

        factorial_h[i] = factorial_h[i-1] * i;

    }

    cudaMemcpyToSymbol(factorial, factorial_h, 65*sizeof(double));

}


__device__ __host__ uint64_t nCr(int32_t n, int32_t r) {

    #ifdef __CUDA_ARCH__

        return (uint64_t) (factorial[n] / (factorial[r] * factorial[n - r]));

    #else

        return (uint64_t) (factorial_h[n] / (factorial_h[r] * factorial_h[n - r]));

    #endif


}

