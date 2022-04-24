
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "set.h"

set_t* make_set() {

    set_t* s = (set_t*) malloc(sizeof(set_t));

    s->vals = NULL;
    s->size = 0;

    return s;

}

void set_insert(set_t* set, uint32_t x) {

    if(set->size == 0) {

        set->vals = (uint32_t*) malloc( sizeof(uint32_t) );
        set->vals[0] = x;
        set->size++;
        
    } else {

        for(uint32_t i=0; i<set->size; i++) {

            if(set->vals[i] == x) {
                return;
            }

        }

        set->vals = (uint32_t*) realloc(set->vals, sizeof(uint32_t) * set->size + 1);
        set->vals[set->size] = x;
        set->size++;

    }

}

uint32_t find_position(set_t* X, uint32_t element) {

    for(uint32_t i=0; i < X->size; i++) {
        
        if (X->vals[i] == element) {
            return i;
        }
    
    }

    return -1; 

}

int element_exists(uint32_t element, set_t* set, uint64_t mask) {

    for(uint32_t i=0; i<set->size; i++) {

        if(element == set->vals[i] && ((mask >> (set->size - i - 1)) & 1) == 1) {
            return 1;
        }

    }

    return 0;

}
    

void print_set(set_t* X) {

    printf("{");

    for(uint32_t i=0; i<X->size; i++) {

        printf("%d", X->vals[i]);

        if(i < X->size - 1) {
            printf(", ");
        }

    }

    printf("}\n");

}

void destroy_set(set_t* set) {

    free(set->vals);
    set->vals = NULL;
    set->size = 0;   
    free(set);

}
