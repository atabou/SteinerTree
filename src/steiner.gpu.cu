
#include <stdio.h>
#include <stdlib.h>
#include <float.h>


#include "steiner.h"
#include "combination.h"
#include "util.h"


#define BLOCK_2D_SIZE 32
#define BLOCK_1D_SIZE 1024
#define MAX_BLOCKS 65536

#define BLOCK_SIZE 1024


__global__ void dw_fill_base_cases(cudatable::table_t* costs, cudagraph::graph_t* g, cudatable::table_t* distances, cudaquery::query_t* terminals) {

    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < terminals->size * costs->m) {
        
        int32_t v = thread_id % costs->m;
        uint64_t mask = 1llu << (thread_id / costs->m);

        int32_t u = terminals->vals[terminals->size - __ffsll(mask)];

        costs->vals[(mask - 1) * costs->m + v] = distances->vals[v * distances->m + u];

    }

}


__global__ void dw_fill_kth_combination(cudatable::table_t* costs, cudagraph::graph_t* g, cudatable::table_t* distances, cudaquery::query_t* terminals, int32_t k) {

    // Assign the position in the table that the block will update.

    int32_t v = blockIdx.x;
    int32_t i = blockIdx.y;

    // Declare and initialize minimum shared memory array.

    __shared__ float minimums[BLOCK_SIZE];
    
    minimums[threadIdx.x] = FLT_MAX;

    // Compute the mask of the current block

    uint64_t mask = ith_combination(terminals->size, k, i);

    // Compute the minimum steiner tree

    for(int32_t w = threadIdx.x; w<gridDim.x; w+=blockDim.x) {

        int32_t exists = 0;
        int32_t position = 0;

        for(int32_t i=0; i<terminals->size; i++) {

            if(w == terminals->vals[i] && ((mask >> (terminals->size - i - 1)) & 1) == 1 ) {

                exists = 1;
                position = i;

            }

            __syncthreads();

        }

        if(exists) {

            uint64_t submask = 1ll << (terminals->size - position - 1);

            float cost = distances->vals[v * distances->m + w] 
                           + costs->vals[((mask & ~submask) - 1) * costs->m + w];

            minimums[threadIdx.x] = (cost < minimums[threadIdx.x]) ? cost : minimums[threadIdx.x];

        } else if(g->deg[w] >= 3) {

            for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) {

                float cost = distances->vals[v * distances->m + w]
                               + costs->vals[(submask - 1) * costs->m + w]
                               + costs->vals[((mask & ~submask) - 1) * costs->m + w];

                minimums[threadIdx.x] = (cost < minimums[threadIdx.x]) ? cost : minimums[threadIdx.x]; //TODO: Do I need to syncthreads here?

            }

        }

        __syncthreads();

    }

    // Parallel minimum reduction

    for(int i=2; i<BLOCK_SIZE; i*=2) {

        if(threadIdx.x % i == 0) {

            minimums[threadIdx.x] = (minimums[threadIdx.x] < minimums[threadIdx.x + i / 2]) ? minimums[threadIdx.x] : minimums[threadIdx.x + i / 2];

        }

    }

    // Assign the minimum to the corresponding slot in the table.

    if(threadIdx.x == 0) {
    
        costs->vals[(mask - 1) * costs->m + v] = minimums[0];
    
    }

}


/**
 * Works for any values of T and V satisfying the following equation 2^T * V < 2^26
 * This could be improved to 2^T * V < 2^58
 */
void base_case(cudatable::table_t* table, cudagraph::graph_t* g, int32_t g_size, cudaquery::query_t* t, int32_t t_size, cudatable::table_t* distances) {

    uint64_t num_thread = g_size * t_size;
    uint64_t num_blocks = (num_thread + BLOCK_1D_SIZE - 1) / BLOCK_1D_SIZE;

    dw_fill_base_cases<<<num_blocks, BLOCK_1D_SIZE>>>(table, g, distances, t);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Base case exit. (Error code: %d)\n", err);
        exit(err);
    }


}


void fill_kth_combination(cudatable::table_t* table, cudagraph::graph_t* g, int32_t g_size, cudaquery::query_t* t, int32_t t_size, cudatable::table_t* distances, int32_t k) {

    initialize_factorial_table();

    dim3 num_threads_per_block(BLOCK_SIZE);

    int64_t num_blocks_x = g_size;
    int64_t num_blocks_y = nCr(t_size, k);

    dim3 num_blocks(num_blocks_x, num_blocks_y);

    dw_fill_kth_combination<<<num_blocks, num_threads_per_block>>>(table, g, distances, t, k);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
        exit(err);
    }

}


void fill_steiner_tree_cuda_table(cudatable::table_t* table, cudagraph::graph_t* g, int32_t g_size, cudaquery::query_t* t, int32_t t_size, cudatable::table_t* distances) {

    // Calculate base case

    base_case(table, g, g_size, t, t_size, distances);

    // Fill table by multiple subsequent kernel calls

    for(int32_t k=2; k <= t_size; k++) {

        fill_kth_combination(table, g, g_size, t, t_size, distances, k);

    }

}


void steiner_tree_gpu(cudagraph::graph_t* graph, int32_t nvrt, cudaquery::query_t* terminals, int32_t nterm, cudatable::table_t* distances, steiner_result** result) {

    // Declare required variables

    table::table_t* costs = NULL;
    cudatable::table_t* costs_d = NULL;

    // Construct the costs table.

    cudatable::make(&costs_d, (int32_t) pow(2, nterm) - 1, nvrt);

    // Fill the costs table.

    TIME(fill_steiner_tree_cuda_table(costs_d, graph, nvrt, terminals, nterm, distances), "\tDW GPU:");

    // Get the filled table from the GPU.

    cudatable::transfer_from_gpu(&costs, costs_d);

    // Initialize steiner result structure

    *result = (steiner_result*) malloc(sizeof(steiner_result));

    // Extract the minimum from the table.

    (*result)->cost = FLT_MAX;
    
    for(int32_t i=0; i < costs->m; i++) {

        if(costs->vals[costs->m *(costs->n - 1) + i] < (*result)->cost) {
            (*result)->cost = costs->vals[costs->m *(costs->n - 1) + i];
        }

    }

    // Free

    cudatable::destroy(costs_d);
    table::destroy(costs);

}