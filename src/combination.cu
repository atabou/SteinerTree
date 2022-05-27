
#include <stdio.h>

#include "combination.h"

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

__device__ __host__ int next_combination(uint32_t n, uint32_t k, uint64_t* mask) {

    if(*mask == 0) {

        *mask = (1ll << k) - 1ll;

    } else {

        uint64_t c = (*mask) & -(*mask);
        uint64_t r = (*mask) + c;
        *mask = r | (((r ^ (*mask)) >> 2)/c);

    }

    return *mask <= (1ll << n) - (1ll << (n-k));

}
