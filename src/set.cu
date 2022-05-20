
#include <stdio.h>

extern "C" {
    #include "set.cuda.h"
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



