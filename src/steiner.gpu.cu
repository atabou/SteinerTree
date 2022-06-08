
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "steiner.hpp"
#include "combination.hpp"
#include "util.hpp"

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536


__global__ void dw_fill_base_cases(table::table_t<float>* costs, table::table_t<int32_t>* roots, table::table_t<int64_t>* trees, graph::graph_t* g, table::table_t<float>* distances, cudaquery::query_t* terminals) {

    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < terminals->size * costs->m) {
        
        int32_t v = thread_id % costs->m;
        uint64_t mask = 1llu << (thread_id / costs->m);

        int32_t u = terminals->vals[terminals->size - __ffsll(mask)];

        costs->vals[(mask - 1) * costs->m + v] = distances->vals[v * distances->m + u];
        roots->vals[(mask - 1) * costs->m + v] = u;
        trees->vals[(mask - 1) * costs->m + v] = -1;

    }

}


__global__ void dw_fill_kth_combination(table::table_t<float>* costs, table::table_t<int32_t>* roots, table::table_t<int64_t>* trees, graph::graph_t* g, table::table_t<float>* distances, cudaquery::query_t* terminals, int32_t k) {

    // Assign the position in the table that the block will update.

    int32_t v = blockIdx.x;
    int32_t i = blockIdx.y;

    // Declare and initialize minimum shared memory array.

    __shared__ float   costs_s[BLOCK_SIZE];
    __shared__ int32_t roots_s[BLOCK_SIZE];
    __shared__ int64_t trees_s[BLOCK_SIZE];

    costs_s[threadIdx.x] = FLT_MAX;
    roots_s[threadIdx.x] = -1;
    trees_s[threadIdx.x] = -1;

    // Compute the mask of the current block

    uint64_t mask = ith_combination(terminals->size, k, i);

    __syncthreads();

    // Compute the minimum steiner tree

    if(threadIdx.x < gridDim.x) {

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

                if(cost < costs_s[threadIdx.x]) {

                    costs_s[threadIdx.x] = cost;
                    roots_s[threadIdx.x] = w;
                    trees_s[threadIdx.x] = mask & ~submask;

                }

            } else if(g->deg[w] >= 3) {

                for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) {

                    float cost = distances->vals[v * distances->m + w]
                                + costs->vals[(submask - 1) * costs->m + w]
                                + costs->vals[((mask & ~submask) - 1) * costs->m + w];

                    if(cost < costs_s[threadIdx.x]) {

                        costs_s[threadIdx.x] = cost;
                        roots_s[threadIdx.x] = w;
                        trees_s[threadIdx.x] = submask;

                    }

                    //TODO: Do I need to syncthreads here?


                }

            }

            __syncthreads();

        }

        // Parallel minimum reduction

        for(int i=2; i<BLOCK_SIZE; i*=2) {

            if(threadIdx.x % i == 0) {

                if(costs_s[threadIdx.x + i / 2] < costs_s[threadIdx.x]) {

                    costs_s[threadIdx.x] = costs_s[threadIdx.x + i / 2];
                    roots_s[threadIdx.x] = roots_s[threadIdx.x + i / 2];
                    trees_s[threadIdx.x] = trees_s[threadIdx.x + i / 2];

                }

            }

            __syncthreads();

        }

        // Assign the minimum to the corresponding slot in the table.

        if(threadIdx.x == 0) {
        
            costs->vals[(mask - 1) * costs->m + v] = costs_s[0];
            roots->vals[(mask - 1) * costs->m + v] = roots_s[0]; 
            trees->vals[(mask - 1) * costs->m + v] = trees_s[0];
        
        }

    }

}


/**
 * Works for any values of T and V satisfying the following equation 2^T * V < 2^26
 * This could be improved to 2^T * V < 2^58
 */
void base_case(cudatable::table_t<float>* costs, cudatable::table_t<int32_t>* roots, cudatable::table_t<int64_t>* trees, cudagraph::graph_t* graph, cudaquery::query_t* t, int32_t t_size, cudatable::table_t<float>* distances) {

    uint64_t num_thread = graph->vrt * t_size;
    uint64_t num_blocks = (num_thread + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dw_fill_base_cases<<<num_blocks, BLOCK_SIZE>>>(costs->table, roots->table, trees->table, graph->graph, distances->table, t);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Base case exit. (Error code: %d)\n", err);
        exit(err);
    }


}


void fill_kth_combination(cudatable::table_t<float>* costs, cudatable::table_t<int32_t>* roots, cudatable::table_t<int64_t>* trees, cudagraph::graph_t* graph, cudaquery::query_t* t, int32_t t_size, cudatable::table_t<float>* distances, int32_t k) {

    initialize_factorial_table();

    dim3 num_threads_per_block(BLOCK_SIZE);

    int64_t num_blocks_x = graph->vrt;
    int64_t num_blocks_y = nCr(t_size, k);

    dim3 num_blocks(num_blocks_x, num_blocks_y);

    dw_fill_kth_combination<<<num_blocks, num_threads_per_block>>>(costs->table, roots->table, trees->table, graph->graph, distances->table, t, k);

    cudaError_t err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
        exit(err);
    }

}


void fill_steiner_tree_cuda_table(cudatable::table_t<float>* costs, cudatable::table_t<int32_t>* roots, cudatable::table_t<int64_t>* trees, cudagraph::graph_t* graph, cudaquery::query_t* t, int32_t t_size, cudatable::table_t<float>* distances) {

    // Calculate base case

    base_case(costs, roots, trees, graph, t, t_size, distances);

    // Fill table by multiple subsequent kernel calls

    for(int32_t k=2; k <= t_size; k++) {

        fill_kth_combination(costs, roots, trees, graph, t, t_size, distances, k);

    }

}




void steiner_tree_gpu(cudagraph::graph_t* graph, cudaquery::query_t* terminals, int32_t nterm, cudatable::table_t<float>* distances, table::table_t<int32_t>* predecessors, steiner::result_t** result) {

    // Declare required variables

    cudatable::table_t< float >* costs_d = NULL;
    cudatable::table_t<int32_t>* roots_d = NULL;
    cudatable::table_t<int64_t>* trees_d = NULL;

    // Construct the results table for the GPU.

    cudatable::make(&costs_d, (int32_t) pow(2, nterm) - 1, graph->vrt);
    cudatable::make(&roots_d, (int32_t) pow(2, nterm) - 1, graph->vrt);
    cudatable::make(&trees_d, (int32_t) pow(2, nterm) - 1, graph->vrt);

    // Fill the costs table.

    TIME(fill_steiner_tree_cuda_table(costs_d, roots_d, trees_d, graph, terminals, nterm, distances), "\tDW GPU:");

    // Initialize steiner result structure

    *result = (steiner::result_t*) malloc(sizeof(steiner::result_t));

    // Get the filled tables from the GPU.

    cudatable::transfer_from_gpu(&((*result)->costs), costs_d);
    cudatable::transfer_from_gpu(&((*result)->roots), roots_d);
    cudatable::transfer_from_gpu(&((*result)->trees), trees_d);
     
    // Extract the minimum from the table.

    (*result)->cost = FLT_MAX;
    
    for(int32_t i=0; i < (*result)->costs->m; i++) {

        if((*result)->costs->vals[(*result)->costs->m * ((*result)->costs->n - 1) + i] < (*result)->cost) {
            
            (*result)->cost = (*result)->costs->vals[(*result)->costs->m * ((*result)->costs->n - 1) + i];
            (*result)->root = i;
            (*result)->tree = (*result)->costs->n;

        }

    }

    // Free

    cudatable::destroy(trees_d);
    cudatable::destroy(roots_d);
    cudatable::destroy(costs_d);

}
