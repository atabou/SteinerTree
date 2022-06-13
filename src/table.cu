
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "table.hpp"

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536

template <class T>
void table::make(table::table_t<T>** t, int32_t n, int32_t m) {

    *t = (table::table_t<T>*) malloc(sizeof(table::table_t<T>));

    (*t)->n = n;
    (*t)->m = m;

    (*t)->vals = (T*) malloc(sizeof(T) * n * m);

}

template <class T>
__global__ void set_table_values_kernel(table::table_t<T>* table, T val);

template <class T>
void set_table_values(table::table_t<T>* table, int32_t n, int32_t m, T val);

template <class T>
void cudatable::make(cudatable::table_t<T>** table_d, int32_t n, int32_t m) {

    table::table_t<T> tmp;

    tmp.n = n;
    tmp.m = m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(T) * n * m);

    if(err) {
        printf("Could not allocat %llu x %llu cuda table. (Error code: %d)\n", (unsigned long long) n, (unsigned long long) m, err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    // Initialize cuda table

    (*table_d) = (cudatable::table_t<T>*) malloc(sizeof(cudatable::table_t<T>));

    // Set cuda table "meta" information

    (*table_d)->n = n;
    (*table_d)->m = m;

    // Allocate table memory on the GPU and save its address in the cuda table structure

    err = cudaMalloc(&((*table_d)->table), sizeof(table::table_t<T>));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    // Copy the constructed tmp to the saved table memory address

    cudaMemcpy((*table_d)->table, &tmp, sizeof(table::table_t<T>), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Initialize all memory cells to the specified value

    set_table_values((*table_d)->table, n, m, (T) INT32_MAX);

}


template <class T>
void cudatable::transfer_to_gpu(cudatable::table_t<T>** table_d, table::table_t<T>* table) {

    table::table_t<T> tmp;

    tmp.n = table->n;
    tmp.m = table->m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(T) * tmp.n * tmp.m);

    if(err) {
        printf("Could not allocat %d x %d cuda table. (Error code: %d)\n", tmp.n, tmp.m, err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(tmp.vals, table->vals, sizeof(T) * tmp.n * tmp.m, cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Initialize cuda table

    (*table_d) = (cudatable::table_t<T>*) malloc(sizeof(cudatable::table_t<T>));

    // Set cuda table "meta" information

    (*table_d)->n = table->n;
    (*table_d)->m = table->m;

    // Allocate table memory on the GPU and save its address in the cuda table structure
    
    err = cudaMalloc(&((*table_d)->table), sizeof(table::table_t<T>));

    if(err) {
        printf("Could not allocate cuda table. (Error code: %d)\n", err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy((*table_d)->table, &tmp, sizeof(table::table_t<T>), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table memory copy. (Error code: %d)\n", err);
        exit(err);
    }

}


template <class T>
void table::print(table::table_t<T>* table) {
    
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

            if(table->vals[i * table->m + j] == (T) INT32_MAX) {
                printf("\033[0;31m%3d\033[0m|", -1);
            } else {
                printf("%.1f|", (float) table->vals[i * table->m + j]);
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


template <class T>
void table::destroy(table::table_t<T>* t) {

    free(t->vals);

    t->vals = NULL;
    t->n = 0;
    t->m = 0;

    free(t);
    
}


// Helpers

template <class T>
__global__ void set_table_values_kernel(table::table_t<T>* table, T val) {

    int32_t pos =  blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
				 + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
				 + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
				 + threadIdx.x; // The position of the thread in the block

    if(pos < table->m * table->n) {
        table->vals[pos] = val;
    }

}


template <class T>
void set_table_values(table::table_t<T>* table, int32_t n, int32_t m, T val) {

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


template <class T>
void cudatable::transfer_from_gpu(table::table_t<T>** table, cudatable::table_t<T>* table_d) {
    
    cudaError_t err;

    *table = (table::table_t<T>*) malloc(sizeof(table::table_t<T>));

    cudaMemcpy(*table, table_d->table, sizeof(table::table_t<T>), cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy cuda table data before free. (Error code: %d).\n", err);
        exit(err);
    }

    T* tmp = (T*) malloc(sizeof(T) * (*table)->m * (*table)->n);
    
    cudaMemcpy(tmp, (*table)->vals, sizeof(T) * (*table)->n * (*table)->m, cudaMemcpyDeviceToHost);

    (*table)->vals = tmp;

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete cuda table deallocation. (Error code: %d)\n", err);
        exit(err);
    }

}


template <class T>
void cudatable::destroy(cudatable::table_t<T>* cuda_table) {

    cudaError_t err;
    table::table_t<T> tmp;

    cudaMemcpy(&tmp, cuda_table->table, sizeof(table::table_t<T>), cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy cuda table data before free. (Error code: %d).\n", err);
        exit(err);
    }

    err = cudaFree(cuda_table->table);

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

    free(cuda_table);

}

template void table::make(table::table_t<int32_t>** t, int32_t n, int32_t m);
template void table::make(table::table_t<int64_t>** t, int32_t n, int32_t m);
template void table::make(table::table_t<float>** t, int32_t n, int32_t m);

template void cudatable::make(cudatable::table_t<float>** table_d, int32_t n, int32_t m);
template void cudatable::make(cudatable::table_t<int32_t>** table_d, int32_t n, int32_t m);
template void cudatable::make(cudatable::table_t<int64_t>** table_d, int32_t n, int32_t m);

template void cudatable::transfer_to_gpu(cudatable::table_t<int32_t>** table_d, table::table_t<int32_t>* table);
template void cudatable::transfer_to_gpu(cudatable::table_t<int64_t>** table_d, table::table_t<int64_t>* table);
template void cudatable::transfer_to_gpu(cudatable::table_t<float>** table_d, table::table_t<float>* table);

template void cudatable::transfer_from_gpu(table::table_t<int32_t>** table, cudatable::table_t<int32_t>* table_d);
template void cudatable::transfer_from_gpu(table::table_t<int64_t>** table, cudatable::table_t<int64_t>* table_d);
template void cudatable::transfer_from_gpu(table::table_t<float>** table, cudatable::table_t<float>* table_d);

template void table::print(table::table_t<int32_t>* table);
template void table::print(table::table_t<int64_t>* table);
template void table::print(table::table_t<float>* table);

template void table::destroy(table::table_t<int32_t>* t);
template void table::destroy(table::table_t<int64_t>* t);
template void table::destroy(table::table_t<float>* t);

template void cudatable::destroy(cudatable::table_t<int32_t>* cuda_table);
template void cudatable::destroy(cudatable::table_t<int64_t>* cuda_table);
template void cudatable::destroy(cudatable::table_t<float>* cuda_table);
