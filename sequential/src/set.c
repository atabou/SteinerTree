
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "set.h"

struct set_t {

    int* elements;
    int size;

};

int get_element(set_t* X, int i) {

    return X->elements[i];

}

int set_size(set_t* X) {

    return X->size;

}

int element_exists(int element, set_t* X) {

    for(int i=0; i<X->size; i++) {

        if(element == X->elements[i]) {
            return 1;
        }

    }

    return 0;

}
    
set_t* remove_element(int element, set_t* X) {

    set_t* Y = (set_t*) malloc( sizeof(set_t) );

    Y->elements = (int*) malloc(sizeof(int) * (X->size - 1));
    Y->size = X->size - 1;

    int flag = 0;

    for(int i=0; i<Y->size; i++) {

        if(X->elements[i] == element) {
            flag = 1;
        }

        if(flag == 0) {
            Y->elements[i] = X->elements[i];
        } else {
            Y->elements[i] = X->elements[i+1];
        }

    }

    return Y;

}

set_t* get_subset(set_t* X, long long mask) {

    set_t* subset = (set_t*) malloc(sizeof(set_t)); 
    
    subset->elements = (int*) malloc( sizeof(int) * __builtin_popcount(mask) );
    subset->size = __builtin_popcount(mask);

    int count = 0;
    
    long long submask = mask;

    while(submask != 0) {

        int pos = X->size - __builtin_ctz(mask) - 1; // Get the position of the first 1 in the bit mask and convert it to the relative position in the set.
        
        subset->elements[count] = X->elements[pos];
        count++;

        submask = ((submask >> __builtin_ctz(submask)) & ~(1LL)) << __builtin_ctz(submask); // Removes the first 1 in the bit mask

    }

    return subset;

}