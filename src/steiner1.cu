
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    #include "steiner1.cuda.h"
}

#include "combination.cuda.h"

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536

__global__ void dw_kernel_1(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, uint32_t k, uint64_t nCr) {


    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < nCr * costs->n) {

        uint64_t v = thread_id / nCr;
        uint64_t mask = ith_combination(terminals->size, k, thread_id % nCr);

        if(k == 1) {

            uint32_t u = terminals->vals[terminals->size - __ffsll(mask)];
            costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];

        } else {

            for(uint32_t u=0; u < g->vrt; u++) {

                uint32_t exists = 0;
                uint32_t position = 0;

                for(uint32_t i=0; i<terminals->size; i++) {

                    if(u == terminals->vals[i] && ((mask >> (terminals->size - i - 1)) & 1) == 1 ) {

                        exists = 1;
                        position = i;

                    }

                }

                if(exists) {


                    uint64_t submask = 1llu << (terminals->size - position - 1);

                    uint32_t cost = distances->vals[v * distances->m + u]
                                  + costs->vals[u * costs->m + ((mask & ~submask) - 1)];

                    atomicMin(&(costs->vals[v * costs->m + (mask - 1)]), cost);

                } else if(g->deg[u] >= 3) {
 
                    for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                        uint32_t cost = distances->vals[v * distances->m + u] + costs->vals[u * costs->m + submask - 1] + costs->vals[u * costs->m + (mask & ~submask) - 1];

                        atomicMin(&(costs->vals[v * costs->m + (mask - 1)]), cost);

                    }

                }

            }

        }

    }

    /* __syncthreads(); */

    /* if(k == terminals->size && thread_id == 0) { */
    /*     for(uint64_t i = 0; i < costs->n; i++) { */
    /*         for(uint64_t j = 0; j < costs->m; j++) { */

    /*             if(costs->vals[i * costs->m + j] == UINT32_MAX) { */
    /*                 printf("-1 "); */
    /*             } else { */
    /*                 printf("%2d ", costs->vals[i*costs->m + j]); */
    /*             } */
    /*         } */
    /*         printf("\n"); */
    /*     } */
    /* } */

}

/**
 * Only works on |T| < 36 (total number of threads than can be launched is 2^58: 65536^3  * 1024 = (2^16)^3 * 2^10 = 2^48 * 2^10 = 2^58)
 */
void fill_steiner_dp_table_gpu_1(cudatable_t* table, cudagraph_t* g, cudaset_t* t, uint64_t g_size, uint64_t t_size, cudatable_t* distances) {

    // Fill table by multiple subsequent kernel calls

    for(uint32_t k=1; k <= t_size; k++) {

        uint64_t num_threads = nCr(t_size, k) * g_size;
        uint64_t num_blocks =  (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if(num_blocks <= MAX_BLOCKS) {

            dim3 vblock(num_blocks, 1, 1);
            dw_kernel_1<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, nCr(t_size, k));

        } else {

            num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if(num_blocks <= MAX_BLOCKS) {

                dim3 vblock(MAX_BLOCKS, num_blocks, 1);
                dw_kernel_1<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, nCr(t_size, k));

            } else {

                num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if(num_blocks <= MAX_BLOCKS) {

                    dim3 vblock(MAX_BLOCKS, MAX_BLOCKS, num_blocks);
                    dw_kernel_1<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, nCr(t_size, k));


                } else {

                    // TODO handle this case.

                }

            }

        }

        cudaError_t err = cudaDeviceSynchronize();

        if(err) {
            printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
            exit(err);
        }

    }

}

