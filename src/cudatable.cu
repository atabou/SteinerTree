
#include "stdio.h"
#include "cudatable.cuh"

cudatable_t* make_cudatable(uint64_t n, uint64_t m) {

    cudatable_t tmp;

    tmp.n = n;
    tmp.m = m;
 
    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(uint32_t) * n * m);

    if(err) {
        printf("Could not allocat %llu x %llu cuda table. (Error code: %d)\n", n, m, err);
    }

    cudatable_t* cuda_table = NULL;

    err = cudaMalloc(&cuda_table, sizeof(cudatable_t));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
    }

    cudaMemcpy(cuda_table, &tmp, sizeof(cudatable_t), cudaMemcpyHostToDevice);

    return cuda_table;

}

cudatable_t* copy_cuda_table(table_t* cpu_table) {

    cudatable_t tmp;

    tmp.n = cpu_table->n;
    tmp.m = cpu_table->m;
 
    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(uint32_t) * tmp.n * tmp.m);

    if(err) {
        printf("Could not allocat %llu x %llu cuda table. (Error code: %d)\n", tmp.n, tmp.m, err);
    }

    cudaMemcpy(tmp.vals, cpu_table->vals, sizeof(uint32_t) * tmp.n * tmp.m, cudaMemcpyHostToDevice);

    cudatable_t* cuda_table = NULL;

    err = cudaMalloc(&cuda_table, sizeof(cudatable_t));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
    }

    cudaMemcpy(cuda_table, &tmp, sizeof(cudatable_t), cudaMemcpyHostToDevice);

    return cuda_table;

}

void free_cuda_table(cudatable_t* cuda_table) {

    cudatable_t tmp;

    cudaMemcpy(&tmp, cuda_table, sizeof(cudatable_t), cudaMemcpyDeviceToHost);

    cudaError_t err;

    err = cudaFree(cuda_table);
    
    if(err) {
        printf("Could not deallocate cuda table. (Error code: %d)\n", err);
        exit(err);
    } 

    err = cudaFree(tmp.vals);

    if(err) {
        printf("Could not deallocate memory for cuda table. (Error code: %d)\n", err);
        exit(err);
    }

}
