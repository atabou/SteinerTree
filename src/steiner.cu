
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    #include "steiner.cuda.h"
}

#include "combination.cuda.h"

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 65536

/**
 * Only can address 2^58 of the 2^64 possible terminals.
 */
__global__ void dw_power_set(cudatable_t* costs, cudatable_t* distances, uint64_t mask, uint32_t v, uint32_t w) {

	uint64_t pos = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
				 + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
				 + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
				 + threadIdx.x; // The position of the thread in the block

	if(pos < (1llu << __popcll(mask)) - 2) {
	
		uint64_t submask = ith_subset(mask, pos);

		uint32_t cost = distances->vals[v * distances->m + w] + costs->vals[w * costs->m + submask - 1] + costs->vals[w * costs->m + (mask & ~submask) - 1];

        atomicMin(&(costs->vals[v * costs->m + (mask - 1)]), cost);

    }

}


__global__ void dw_recurrence_relation(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, uint64_t mask, uint32_t v) {

    uint32_t w =  blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if(w < costs->n) {

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


            uint64_t submask = 1llu << (terminals->size - position - 1);

            uint32_t cost = distances->vals[v * distances->m + w]
                          + costs->vals[w * costs->m + ((mask & ~submask) - 1)];
           
            atomicMin(&(costs->vals[v * costs->m + (mask - 1)]), cost);

        } else if(g->deg[w] >= 3) {

            uint64_t num_threads = costs->m - 2; // -2 to remove all 0s and all 1s
            uint64_t num_blocks = (num_threads / BLOCK_SIZE) + 1;

            if(num_blocks <= MAX_BLOCKS) {

                dim3 vblock(num_blocks, 1, 1);
                dw_power_set<<<vblock, BLOCK_SIZE>>>(costs, distances, mask, v, w);

            } else {

                num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if(num_blocks <= MAX_BLOCKS) {

                    dim3 vblock(MAX_BLOCKS, num_blocks, 1);	
                    dw_power_set<<<vblock, BLOCK_SIZE>>>(costs, distances, mask, v, w);

                } else {

                    num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    if(num_blocks <= MAX_BLOCKS) {

                        dim3 vblock(MAX_BLOCKS, MAX_BLOCKS, num_blocks);						
                        dw_power_set<<<vblock, BLOCK_SIZE>>>(costs, distances, mask, v, w);

                    } else {

                        // TODO handle this case.

                    }

                }

            }

        }

    }

}

__global__ void dw_combination_kernel(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, uint64_t mask) {

    uint32_t v = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if(v < costs->n) {

        if(__popcll(mask) == 1) {

            uint32_t u = terminals->vals[terminals->size - __ffsll(mask)];
            costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];

        } else {

            uint32_t num_threads = costs->n;
            uint32_t num_blocks = (num_threads / BLOCK_SIZE) + 1;

            if(num_blocks <= MAX_BLOCKS) {

                dim3 vblock(num_blocks, 1, 1);
                dw_recurrence_relation<<<vblock, BLOCK_SIZE>>>(costs, g, distances, terminals, mask, v);

            } else {

                num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE; // No need to check furthure as we have reached the max size of uint32 with 2D grid.

                dim3 vblock(MAX_BLOCKS, num_blocks, 1);
                dw_recurrence_relation<<<vblock, BLOCK_SIZE>>>(costs, g, distances, terminals, mask, v);

            }

        }

    }	

}

__global__ void dw_kernel(cudatable_t* costs, cudagraph_t* g, cudatable_t* distances, cudaset_t* terminals, uint32_t k, uint64_t nCr) {


    uint64_t thread_id = blockIdx.z * gridDim.y * gridDim.x * blockDim.x // Number of threads inside the 3D part of the grid coming before the thread in question.
        + blockIdx.y * gridDim.x * blockDim.x // Number of threads inside the 2D part of the grid coming before the thread in question.
        + blockIdx.x * blockDim.x  // Number of threads inside the 1D part of the grid coming before the thread in question.
        + threadIdx.x; // The position of the thread in the block

    if(thread_id < nCr) {

        uint64_t mask = ith_combination(terminals->size, k, thread_id);

        uint32_t num_threads = costs->n;
        uint32_t num_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if(num_blocks <= MAX_BLOCKS) {

            dim3 vblocks(num_blocks, 1, 1);
            dw_combination_kernel<<<vblocks, BLOCK_SIZE>>>(costs, g, distances, terminals, mask);

        } else {

            num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE; // No need to check furthure as we have reached the max size of uint32 with 2D grid.

            dim3 vblocks(MAX_BLOCKS, num_blocks, 1);
            dw_combination_kernel<<<vblocks, BLOCK_SIZE>>>(costs, g, distances, terminals, mask);

        }

    }

}

/**
 * Only works on |T| < 58 (total number of threads than can be launched is 2^58: 65536^3  * 1024 = (2^16)^3 * 2^10 = 2^48 * 2^10 = 2^58)
 */
void fill_steiner_dp_table_gpu(cudatable_t* table, cudagraph_t* g, cudaset_t* t, uint32_t t_size, cudatable_t* distances) {

    // Fill table by multiple subsequent kernel calls

    for(uint32_t k=1; k <= t_size; k++) {

        uint64_t num_threads = nCr(t_size, k);
        uint64_t num_blocks =  (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if(num_blocks <= MAX_BLOCKS) {

            dim3 vblock(num_blocks, 1, 1);
            dw_kernel<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, num_threads);

        } else {

            num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if(num_blocks <= MAX_BLOCKS) {

                dim3 vblock(MAX_BLOCKS, num_blocks, 1);
                dw_kernel<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, num_threads);

            } else {

                num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

                if(num_blocks <= MAX_BLOCKS) {

                    dim3 vblock(MAX_BLOCKS, MAX_BLOCKS, num_blocks);
                    dw_kernel<<<vblock, BLOCK_SIZE>>>(table, g, distances, t, k, num_threads);

                } else {

                    // TODO handle this case.

                }

            }

        }

        cudaError_t err = cudaDeviceSynchronize();

        printf("\n");

        if(err) {
            printf("Could not complete the steiner tree call with k: %d. (Error code: %d)\n", k, err);
            exit(err);
        }

    }

}

