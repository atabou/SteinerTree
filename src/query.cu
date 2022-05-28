#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "query.h"


void query::make(query::query_t** s) {

    *s = (query::query_t*) malloc(sizeof(query::query_t));

    (*s)->vals = NULL;
    (*s)->size = 0;

}


void query::insert(query::query_t* set, int32_t x) {

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


int32_t query::find_position(query::query_t* X, int32_t element) {

    for(int32_t i=0; i < X->size; i++) {

        if (X->vals[i] == element) {
            return i;
        }

    }

    return -1;

}


int query::element_exists(int32_t element, query::query_t* set, uint64_t mask) {

    for(int32_t i=0; i<set->size; i++) {

        if(element == set->vals[i] && ((mask >> (set->size - i - 1)) & 1) == 1) {
            return 1;
        }

    }

    return 0;

}


void query::print(query::query_t* X) {

    printf("{");

    for(int32_t i=0; i<X->size; i++) {

        printf("%d", X->vals[i]);

        if(i < X->size - 1) {
            printf(", ");
        }

    }

    printf("}\n");

}


void cudaquery::transfer_to_gpu(cudaquery::query_t** set_d, query::query_t* set) {

    cudaquery::query_t tmp;

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

    err = cudaMalloc(set_d, sizeof(cudaquery::query_t));

    if(err) {
        printf("Could not initialize cuda set. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(*set_d, &tmp, sizeof(cudaquery::query_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda set element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

}


void cudaquery::destroy(cudaquery::query_t* set) {

    cudaquery::query_t tmp;

    cudaMemcpy(&tmp, set, sizeof(cudaquery::query_t), cudaMemcpyDeviceToHost);

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


void query::destroy(query::query_t* set) {

    free(set->vals);
    set->vals = NULL;
    set->size = 0;
    free(set);

}


