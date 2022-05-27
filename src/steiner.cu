
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#include "steiner.h"
#include "combination.h"
#include "util.h"

void fill_steiner_dp_table_cpu(table_t* costs, graph_t* g, set_t* terminals, table_t* distances) {

    for(int32_t k=1; k <= terminals->size; k++) {

        uint64_t mask = 0;

        while( next_combination(terminals->size, k, &mask) ) { // while loop runs T choose k times (nCr).

            for(int32_t v=0; v < costs->n; v++) {

                if(k == 1) {

                    int32_t u = terminals->vals[terminals->size - __builtin_ffsll(mask)];
                    costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];
                    
                } else {
                    
                    float min = FLT_MAX;
                    
                    for(int32_t w=0; w < costs->n; w++) { // O(T * 2^T * (V+E))

                        if( element_exists(w, terminals, mask) ) { // O(V + E)

                            uint64_t submask = 1ll << (terminals->size - find_position(terminals, w) - 1);

                            float cost = distances->vals[v * distances->m + w] 
                                       + costs->vals[w * costs->m + ((mask & ~submask) - 1)];

                            if(cost < min) {

                                min = cost;

                            }

                        } else if(g->deg[w] >= 3) { // O(2^T (V+E))

                            for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                                float cost = distances->vals[v * distances->m + w] 
                                           + costs->vals[w * costs->m + submask - 1] 
                                           + costs->vals[w * costs->m + (mask & ~submask) - 1];

                                if(cost < min) {

                                    min = cost;

                                }

                            }

                        }

                    }

                    costs->vals[v * costs->m + mask - 1] = min;

                }
                
            }

        }

    }

}

steiner_result steiner_tree_cpu(graph_t* g, set_t* terminals, table_t* distances) {

    // Initialize DP table.
    
    table_t* costs = make_table(g->vrt, (int32_t) pow(2, terminals->size) - 1);

    // Fill dp table.

    fill_steiner_dp_table_cpu(costs, g, terminals, distances);

    // Initialize result structure

    steiner_result result;

    // Calculate minimum from table.

    result.cost = FLT_MAX;

    for(int32_t i=costs->m - 1; i < costs->m * distances->n; i+=costs->m) {
        
        if(costs->vals[i] < result.cost) {
            result.cost = costs->vals[i];
        }

    }

    // Free table
   
    free(costs);

    return result;

}

#define BLOCK_2D_SIZE 32
#define BLOCK_1D_SIZE 1024
#define MAX_BLOCKS 65536

__global__ void dw_fill_base_cases(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals) {

    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < terminals->size * costs->n) {

        int32_t v = thread_id / terminals->size;
        uint64_t mask = 1llu << (thread_id % terminals->size);

        // uint64_t mask = ith_combination(terminals->size, 1, thread_id % terminals->size);
        
        int32_t u = terminals->vals[terminals->size - __ffsll(mask)];
        
        costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];

    }

}


__global__ void dw_fill_kth_combination(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, int32_t k) {

    int32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t w = blockDim.y * blockIdx.y + threadIdx.y;

    if(v < g->vrt && w < g->vrt) {
        
        uint64_t mask = 0;

        while( next_combination(terminals->size, k, &mask) ) {

            int32_t exists = 0;
            int32_t position = 0;

            for(int32_t i=0; i<terminals->size; i++) {

                if(w == terminals->vals[i] && ((mask >> (terminals->size - i - 1)) & 1) == 1 ) {

                    exists = 1;
                    position = i;

                }

            }

            __syncthreads();


            if(exists) {

                uint64_t submask = 1ll << (terminals->size - position - 1);

                float cost = distances->vals[v * distances->m + w] 
                           + costs->vals[w * costs->m + ((mask & ~submask) - 1)];

                atomicMin(&costs->vals[v * costs->m + mask - 1], cost);

            } else if(g->deg[w] >= 3) {

                for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                    float cost = distances->vals[v * distances->m + w]
                               + costs->vals[w * costs->m + submask - 1]
                               + costs->vals[w * costs->m + (mask & ~submask) - 1];

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
void base_case(cudatable_t* table, cudagraph_t* g, int32_t g_size, cudaset_t* t, int32_t t_size, cudatable_t* distances) {

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
void fill_kth_combination(cudatable_t* table, cudagraph_t* g, int32_t g_size, cudaset_t* t, int32_t t_size, cudatable_t* distances, int32_t k) {

    int32_t num_thread_x = g_size;
    int32_t num_thread_y = g_size;

    int32_t num_blocks_x = (num_thread_x + BLOCK_2D_SIZE - 1) / BLOCK_2D_SIZE;
    int32_t num_blocks_y = (num_thread_y + BLOCK_2D_SIZE - 1) / BLOCK_2D_SIZE;

    dim3 num_thread_per_block(BLOCK_2D_SIZE, BLOCK_2D_SIZE);
    dim3 num_blocks(num_blocks_x, num_blocks_y);

    dw_fill_kth_combination<<<num_blocks, num_thread_per_block>>>(table, g, distances, t, k);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
        exit(err);
    }


}


void fill_steiner_tree_cuda_table(cudatable_t* table, cudagraph_t* g, int32_t g_size, cudaset_t* t, int32_t t_size, cudatable_t* distances) {

    base_case(table, g, g_size, t, t_size, distances);

    // Fill table by multiple subsequent kernel calls

    for(int32_t k=2; k <= t_size; k++) {

        fill_kth_combination(table, g, g_size, t, t_size, distances, k);

    }

}

steiner_result steiner_tree_gpu(cudagraph_t* graph, int32_t nvrt, cudaset_t* terminals, int32_t nterm, cudatable_t* distances) {

    // Construct the costs table.

    cudatable_t* costs = make_cudatable(nvrt, (int32_t) pow(2, nterm) - 1);

    // Fill the costs table.

    fill_steiner_tree_cuda_table(costs, graph, nvrt, terminals, nterm, distances);

    // Get the filled table from the GPU.

    table_t* result = get_table_from_gpu(costs);

    // Initialize result structure

    steiner_result res;

    // Extract the minimum from the table.

    res.cost = FLT_MAX;
    
    for(int32_t i=result->m - 1; i < result->m * result->n; i = i + result->m) {
       
        if(result->vals[i] < res.cost) {
            res.cost = result->vals[i];
        }

    }

    // Free

    free_cudatable(costs);
    free_table(result);

    return res;

}
