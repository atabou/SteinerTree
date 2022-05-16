
#include <stdio.h>

#include "combination.cuda.h"

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

		if(n > r) {
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

/**
 * O(n^2 + n) - ~ 64*64 + 64
 */
__device__ uint64_t ith_subset(uint64_t mask, uint64_t i) {

    uint64_t lim = 0;

    for(uint32_t comb=1; comb < __popcll(mask); comb++) {

        uint64_t ncr = nCr(__popcll(mask), comb);

        if(i < lim + ncr) { 
    
            uint64_t submask = 0;
            uint64_t subcomb = ith_combination(__popcll(mask), comb, i - lim);

            while(subcomb != 0) {

                uint32_t pos = __ffsll(subcomb) - 1;

                subcomb = subcomb >> pos;

                if((subcomb & 1) == 1) {
                    
                    uint32_t filter = 1llu;
                    uint64_t tmp = mask;

                    for(uint32_t nshifts = 0; nshifts < pos + 1; nshifts++) {
                        
                        filter = (1llu << (__ffsll(tmp) - 1));
                        tmp = tmp >> (__ffsll(tmp)) << (__ffsll(tmp));

                    }

                    submask = submask | filter;

                }

                subcomb = (subcomb >> 1) << (pos + 1);
            
            }

            return submask;

        }

        lim = lim + ncr;

    }

	return 0;

}

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

__device__ int gpu_next_combination(uint32_t n, uint32_t k, uint64_t* mask) {

    if(*mask == 0) {

        *mask = (1ll << k) - 1ll;

    } else {

        uint64_t c = (*mask) & -(*mask);
        uint64_t r = (*mask) + c;
        *mask = r | (((r ^ (*mask)) >> 2)/c);

    }

    return *mask <= (1ll << n) - (1ll << (n-k));

}
