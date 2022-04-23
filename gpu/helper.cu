
#include "common.h"

/**
 * Must return double as factorial(21) > 2^64.
 */
__device__ __host__ double factorial(uint64_t n) {

	double fact = 1;

	for(uint64_t i=1; i<=n; i++) {
		fact *= i;
	}

	return fact;

}

/**
 * Max of nCr is r = n/2 -> 64 C 32 is max and 64 C 32 < 2^64 -> nCr can fit in 64 bit integer for n = 64.
 */
__device__ __host__ uint64_t nCr(uint64_t n, uint64_t r) {

	return (uint64_t) (factorial(n) / (factorial(k) * factorial(n - k)));

}

