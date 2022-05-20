
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

void set_insert(set_t* set, int32_t x) {

    if(set->size == 0) {

        set->vals = (int32_t*) malloc( sizeof(int32_t) );
        set->vals[0] = x;
        set->size++;
        
    } else {

        for(int32_t i=0; i<set->size; i++) {

            if(set->vals[i] == x) {
                return;
            }

        }

        set->vals = (int32_t*) realloc(set->vals, sizeof(int32_t) * set->size + 1);
        set->vals[set->size] = x;
        set->size++;

    }

}

int32_t find_position(set_t* X, int32_t element) {

    for(int32_t i=0; i < X->size; i++) {
        
        if (X->vals[i] == element) {
            return i;
        }
    
    }

    return -1; 

}

int element_exists(int32_t element, set_t* set, uint64_t mask) {

    for(int32_t i=0; i<set->size; i++) {

        if(element == set->vals[i] && ((mask >> (set->size - i - 1)) & 1) == 1) {
            return 1;
        }

    }

    return 0;

}
    

void print_set(set_t* X) {

    printf("{");

    for(int32_t i=0; i<X->size; i++) {

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
