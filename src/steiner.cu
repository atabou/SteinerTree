
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    #include "steiner.cuda.h"
}

#include "combination.cuda.h"
#include "util.h"

#define BLOCK_2D_SIZE 32
#define BLOCK_1D_SIZE 1024
#define MAX_BLOCKS 65536

__device__ void print_cuda_table(cudatable_t* t) {

    for(int v=0; v < t->n; v++) {
        for(int i=0; i<t->m; i++) {
            printf("%d ", t->vals[v * t->m + i]);
        }
        printf("\n");
    }

}

__global__ void dw_fill_base_cases(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals) {

    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < terminals->size * costs->n) {

        uint64_t v = thread_id / terminals->size;
        uint64_t mask = ith_combination(terminals->size, 1, thread_id % terminals->size);

        uint32_t u = terminals->vals[terminals->size - __ffsll(mask)];
        costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];

    }

}

__global__ void dw_fill_kth_combination(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, uint32_t k) {

    uint64_t v = gridDim.x * blockIdx.x + threadIdx.x;
    uint64_t w = gridDim.y * blockIdx.y + threadIdx.y;
 
    if(v < g->vrt && w < g->vrt) {
        
        uint64_t mask = 0;

        while( gpu_next_combination(terminals->size, k, &mask) ) {

            uint32_t exists = 0;
            uint32_t position = 0;

            for(uint32_t i=0; i<terminals->size; i++) {

                if(w == terminals->vals[i] && ((mask >> (terminals->size - i - 1)) & 1) == 1 ) {

                    exists = 1;
                    position = i;

                }

            }

            __syncthreads();


            if(exists) {

                uint64_t submask = 1ll << (terminals->size - position - 1);

                uint32_t cost = distances->vals[v * distances->m + w] 
                    +     costs->vals[w * costs->m + ((mask & ~submask) - 1)];

                atomicMin(&costs->vals[v * costs->m + mask - 1], cost);

            } else if(g->deg[w] >= 3) {

                for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                    uint32_t cost = distances->vals[v * distances->m + w]
                        +     costs->vals[w * costs->m + submask - 1]
                        +     costs->vals[w * costs->m + (mask & ~submask) - 1];

                    atomicMin(&costs->vals[v * costs->m + mask - 1], cost);

                }

            }

            __syncthreads();

        }

    }

}

/**
 * Works for any values of T and V satisfying the following equation 2^T * V < 2^26
 * This could be improved to 2^T * V < 2^58
 */
void base_case(cudatable_t* table, cudagraph_t* g, uint64_t g_size, cudaset_t* t, uint64_t t_size, cudatable_t* distances) {

    uint64_t num_thread = g_size * t_size;
    uint64_t num_blocks = (num_thread + BLOCK_1D_SIZE - 1) / BLOCK_1D_SIZE;

    dw_fill_base_cases<<<num_blocks, BLOCK_1D_SIZE>>>(table, g, distances, t);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Base case exit. (Error code: %d)\n", err);
        exit(err);
    }


}

/**
 * Works only for values of V < 2^21
 */
void fill_kth_combination(cudatable_t* table, cudagraph_t* g, uint64_t g_size, cudaset_t* t, uint64_t t_size, cudatable_t* distances, uint32_t k) {


    uint64_t num_thread_x = g_size;
    uint64_t num_thread_y = g_size;

    uint64_t num_blocks_x = (num_thread_x + BLOCK_2D_SIZE - 1) / BLOCK_2D_SIZE;
    uint64_t num_blocks_y = (num_thread_y + BLOCK_2D_SIZE - 1) / BLOCK_2D_SIZE;

    dim3 num_thread_per_block(BLOCK_2D_SIZE, BLOCK_2D_SIZE);
    dim3 num_blocks(num_blocks_x, num_blocks_y);

    dw_fill_kth_combination<<<num_blocks, num_thread_per_block>>>(table, g, distances, t, k);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
        exit(err);
    }


}

void steiner_tree_gpu(cudatable_t* table, cudagraph_t* g, uint64_t g_size, cudaset_t* t, uint64_t t_size, cudatable_t* distances) {

    base_case(table, g, g_size, t, t_size, distances);

    // Fill table by multiple subsequent kernel calls

    for(uint32_t k=2; k <= t_size; k++) {

        fill_kth_combination(table, g, g_size, t, t_size, distances, k);

    }

}

