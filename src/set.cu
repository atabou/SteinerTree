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


cudaset_t* copy_cudaset(set_t* set) {

    cudaset_t tmp;

    tmp.size = set->size;
   
    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(int32_t) * set->size);

    if(err) {
        printf("Could not initialize cuda set element array. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(tmp.vals, set->vals, sizeof(int32_t) * set->size, cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set element memory copy. (Error code: %d)\n", err);
        exit(err);
    }
 
    cudaset_t* cuda_set = NULL;

    err = cudaMalloc(&cuda_set, sizeof(cudaset_t));

    if(err) {
        printf("Could not initialize cuda set. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(cuda_set, &tmp, sizeof(cudaset_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    return cuda_set;


}


void free_cudaset(cudaset_t* set) {

    cudaset_t tmp;

    cudaMemcpy(&tmp, set, sizeof(cudaset_t), cudaMemcpyDeviceToHost);

    cudaError_t err;

    err = cudaFree(set);

    if(err) {
        printf("Could not free cuda set. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaFree(tmp.vals);

    if(err) {
        printf("Could not free cuda set elements. (Error code: %d)\n", err);
        exit(err);
    }

}


void free_set(set_t* set) {

    free(set->vals);
    set->vals = NULL;
    set->size = 0;
    free(set);

}


