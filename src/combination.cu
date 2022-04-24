
#include <stdio.h>

#include "combination.cuh"

/**
 * Must return double as factorial(21) > 2^64 which is not enough to calculate factorial of 64.
 */
__device__ __host__ double factorial(uint64_t n) {

	double fact = 1;

	for(uint64_t i=1; i<=n; i++) {
		fact *= i;
	}

	return fact;

}

__device__ __host__ uint64_t nCr(uint64_t n, uint64_t r) {

	return (uint64_t) (factorial(n) / (factorial(r) * factorial(n - r)));

}


__device__ uint64_t ith_combination(uint64_t n, uint64_t r, uint64_t i) {
	
	uint64_t mask = 0llu;

	while(n > 0) { // O(n)

		uint64_t y = 0;

		if(n > r && r >= 0) {
			y = nCr(n-1, r); // O(n)
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

__device__ __host__ void print_mask(uint64_t mask, uint32_t size) {

    uint64_t bit = 1ll << (size - 1);

    while(bit != 0) {

        if(mask & bit) {
            printf("1");
        } else{
            printf("0");
        }

        bit = bit >> 1;

    }


}
