
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "table.h"

#include <float.h>
#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536


table_t* make_table(int32_t n, int32_t m) {

    table_t* t = (table_t*) malloc(sizeof(table_t));

    t->n = n;
    t->m = m;

    t->vals = (float*) malloc(sizeof(float) * n * m);

	return t;

}


__global__ void set_table_values_kernel(cudatable_t* table, float val);
__host__ void set_table_values(cudatable_t* table, int32_t n, int32_t m, float val);

cudatable_t* make_cudatable(int32_t n, int32_t m) {

    cudatable_t tmp;

    tmp.n = n;
    tmp.m = m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(float) * n * m);

    if(err) {
        printf("Could not allocat %llu x %llu cuda table. (Error code: %d)\n", (unsigned long long) n, (unsigned long long) m, err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudatable_t* cuda_table = NULL;

    err = cudaMalloc(&cuda_table, sizeof(cudatable_t));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(cuda_table, &tmp, sizeof(cudatable_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Set the initial memory locations of the table

    set_table_values(cuda_table, n, m, FLT_MAX);

    return cuda_table;

}


cudatable_t* copy_cudatable(table_t* table) {

    cudatable_t tmp;

    tmp.n = table->n;
    tmp.m = table->m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(float) * tmp.n * tmp.m);

    if(err) {
        printf("Could not allocat %d x %d cuda table. (Error code: %d)\n", tmp.n, tmp.m, err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(tmp.vals, table->vals, sizeof(float) * tmp.n * tmp.m, cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    cudatable_t* cuda_table = NULL;

    err = cudaMalloc(&cuda_table, sizeof(cudatable_t));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(cuda_table, &tmp, sizeof(cudatable_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    return cuda_table;

}


__device__ __host__ void print_table(table_t* table) {
    
    printf("\n\033[0;32m    |");

    for(int32_t i=0; i<table->m; i++) {
        printf("%3d|", i);
    }

    printf("\n");
    for(int32_t i=0; i<table->m; i++) {
        printf("+---");
    }
    printf("+---+\033[0m\n");

    for(int32_t i=0; i<table->n; i++) {
        printf("\033[0;32m %3d|\033[0m", i);
        for(int32_t j=0; j<table->m; j++) {

            if(table->vals[i * table->m + j] == FLT_MAX) {
                printf("\033[0;31m%3d\033[0m|", -1);
            } else {
                printf("%.1f|", table->vals[i * table->m + j]);
            }
            
        }
        printf("\n\033[0;32m+---+\033[0m");
        for(int32_t j=0; j<table->m; j++) {
            printf("---+");
        }
        printf("\n");
    }
    printf("\n");

}


void free_table(table_t* t) {

    free(t->vals);

    t->vals = NULL;
    t->n = 0;
    t->m = 0;

    free(t);
    
}

// Helpers

__global__ void set_table_values_kernel(cudatable_t* table, float val) {

    int32_t pos =  blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
				 + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
				 + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
				 + threadIdx.x; // The position of the thread in the block

    if(pos < table->m * table->n) {
        table->vals[pos] = val;
    }

}

__host__ void set_table_values(cudatable_t* table, int32_t n, int32_t m, float val) {

    uint64_t num_threads = n * m;
    uint64_t num_blocks =  (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(num_blocks <= MAX_BLOCKS) {

        dim3 vblock(num_blocks, 1, 1);
        set_table_values_kernel<<<vblock, BLOCK_SIZE>>>(table, val);

    } else {

        num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if(num_blocks <= MAX_BLOCKS) {

            dim3 vblock(MAX_BLOCKS, num_blocks, 1);
            set_table_values_kernel<<<vblock, BLOCK_SIZE>>>(table, val);

        } else {

            num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if(num_blocks <= MAX_BLOCKS) {

                dim3 vblock(MAX_BLOCKS, MAX_BLOCKS, num_blocks);
                set_table_values_kernel<<<vblock, BLOCK_SIZE>>>(table, val);

            } else {

                // TODO handle this case.

            }

        }

    }

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not set initial values of table. (Error code: %d)\n", err);
        exit(err);
    }



}





table_t* get_table_from_gpu(cudatable_t* cuda_table) {
    
    cudaError_t err;

    table_t* result = (table_t*) malloc(sizeof(table_t));

    cudaMemcpy(result, cuda_table, sizeof(cudatable_t), cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy cuda table data before free. (Error code: %d).\n", err);
        exit(err);
    }

    float* tmp = (float*) malloc(sizeof(float) * result->m * result->n);
    
    cudaMemcpy(tmp, result->vals, sizeof(float) * result->n * result->m, cudaMemcpyDeviceToHost);

    result->vals = tmp;

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete cuda table deallocation. (Error code: %d)\n", err);
        exit(err);
    }

    return result;


}

void free_cudatable(cudatable_t* cuda_table) {

    cudaError_t err;
    cudatable_t tmp;

    cudaMemcpy(&tmp, cuda_table, sizeof(cudatable_t), cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy cuda table data before free. (Error code: %d).\n", err);
        exit(err);
    }

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

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete cuda table deallocation. (Error code: %d)\n", err);
        exit(err);
    }

}