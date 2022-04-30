
#include <stdio.h>

extern "C" {
    #include "table.cuda.h"
}

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536

__global__ void set_table_values_kernel(cudatable_t* table, uint32_t val) {

    uint64_t pos = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
				 + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
				 + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
				 + threadIdx.x; // The position of the thread in the block

    if(pos < table->m * table->n) {
        table->vals[pos] = val;
    }

}

void set_table_values(cudatable_t* table, uint64_t n, uint64_t m, uint32_t val) {

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

cudatable_t* make_cudatable(uint64_t n, uint64_t m) {

    cudatable_t tmp;

    tmp.n = n;
    tmp.m = m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(uint32_t) * n * m);

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

    set_table_values(cuda_table, n, m, UINT32_MAX);

    return cuda_table;

}

cudatable_t* copy_cudatable(table_t* cpu_table) {

    cudatable_t tmp;

    tmp.n = cpu_table->n;
    tmp.m = cpu_table->m;

    cudaError_t err;

    err = cudaMalloc(&(tmp.vals), sizeof(uint32_t) * tmp.n * tmp.m);

    if(err) {
        printf("Could not allocat %llu x %llu cuda table. (Error code: %d)\n", (unsigned long long) tmp.n, (unsigned long long) tmp.m, err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda table element memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    cudaMemcpy(tmp.vals, cpu_table->vals, sizeof(uint32_t) * tmp.n * tmp.m, cudaMemcpyHostToDevice);

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

void free_cudatable(cudatable_t* cuda_table) {

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
