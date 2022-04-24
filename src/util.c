

#include <stdio.h>

#include "util.h"

int next_combination(uint32_t n, uint32_t k, uint64_t* mask) {

    if(*mask == 0) {

        *mask = (1ll << k) - 1ll;

    } else {

        uint64_t c = (*mask) & -(*mask);
        uint64_t r = (*mask) + c;
        *mask = r | (((r ^ (*mask)) >> 2)/c);

    }

    return *mask <= (1ll << n) - (1ll << (n-k));

}

void print_bits(uint64_t number, uint32_t size) {

    uint64_t bit = 1ll << (size - 1);

    while(bit != 0) {

        if(number & bit) {
            printf("1");
        } else{
            printf("0");
        }

        bit = bit >> 1;

    }

}
