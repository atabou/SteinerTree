
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "steiner.h"
#include "steiner.cuda.h"
#include "steiner1.cuda.h"
#include "steiner2.cuda.h"

#include "shortestpath.h"
#include "util.h"

void fill_steiner_dp_table_cpu(table_t* costs, graph_t* g, set_t* terminals, table_t* distances) {

    for(uint32_t k=1; k <= terminals->size; k++) {

        uint64_t mask = 0;

        while( next_combination(terminals->size, k, &mask) ) { // while loop runs T choose k times (nCr).

            for(int v=0; v < costs->n; v++) {

                if(k == 1) {

                    uint32_t u = terminals->vals[terminals->size - __builtin_ffsll(mask)];
                    costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];
                    
                } else {
                    
                    uint32_t min = UINT32_MAX;
                    
                    for(int w=0; w < costs->n; w++) { // O(T * 2^T * (V+E))

                        if( element_exists(w, terminals, mask) ) { // O(V + E)

                            uint64_t submask = 1ll << (terminals->size - find_position(terminals, w) - 1);

                            int cost = distances->vals[v * distances->m + w] 
                                     + costs->vals[w * costs->m + ((mask & ~submask) - 1)];

                            if(cost < min) {

                                min = cost;

                            }

                        } else if(g->deg[w] >= 3) { // O(2^T (V+E))

                            for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

//								print_bits(mask, terminals->size); printf(" - ");
//								print_bits(submask, terminals->size); printf(" - ");
//								print_bits(mask & ~submask, terminals->size); printf("\n");

                                uint32_t cost = distances->vals[v * distances->m + w] 
                                              + costs->vals[w * costs->m + submask - 1] 
                                              + costs->vals[w * costs->m + (mask & ~submask) - 1];

                                if(cost < min) {

                                    min = cost;

                                }

                            }

//							printf("\n");

                        }

                    }

                    costs->vals[v * costs->m + mask - 1] = min;

                }
                
            }

        }

    }

}

table_t* steiner_tree(graph_t* g, set_t* terminals) {

    printf("|V|= %u, |T|= %u:\n", g->vrt, terminals->size);

    // Initialize DP table.

    clock_t c = clock();
    table_t* costs = make_table(g->vrt, (uint64_t) pow(2, terminals->size) - 1);
    printf("\tInitialize DP table: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    // All pairs shortest path.

    table_t* distances = make_table(g->vrt, g->vrt);
    table_t* parents = make_table(g->vrt, g->vrt);

    c = clock();
    apsp_gpu_graph(g, distances, parents);
    printf("\tAll Pairs Shortest Path: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    // Fill dp table

    c = clock();
    fill_steiner_dp_table_cpu(costs, g, terminals, distances);
    printf("\tCPU DP Table fill: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    /* print_table(costs); */

    // GPU

    c = clock();
    cudatable_t* c_d = make_cudatable(g->vrt, (uint64_t) pow(2, terminals->size) - 1);  
    printf("\tGPU Table create: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    c = clock();
    cudatable_t* d_d = copy_cudatable(distances);
    printf("\tGPU table copy: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);
    
    c = clock();
    cudagraph_t* g_d = copy_cudagraph(g);
    printf("\tGPU graph copy: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    c = clock();
    cudaset_t* t_d = copy_cudaset(terminals);
    printf("\tGPU set copy: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC); 

    // c = clock();
    // fill_steiner_dp_table_gpu(c_d, g_d, t_d, terminals->size, d_d);
    // printf("\tGPU table fill: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);
  
    // c = clock();
    // fill_steiner_dp_table_gpu_1(c_d, g_d, t_d, g->vrt, terminals->size, d_d);
    // printf("\tGPU table fill 1: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    c = clock();
    fill_steiner_dp_table_gpu_2(c_d, g_d, g->vrt, t_d, terminals->size, d_d);
    printf("\tGPU table fill 2: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    free_cudatable(c_d);
    free_cudatable(d_d);
    free_cudagraph(g_d);
    free_cudaset(t_d);

    // Free table

    free_table(distances);
    free_table(parents);

    return costs;

}
