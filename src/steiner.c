
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "steiner.h"
#include "util.h"
#include "shortestpath.h"
#include "table.h"

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

    c = clock();
    pair_t* apsp = all_pairs_shortest_path(g);
    printf("\tAll Pairs Shortest Path: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    table_t* parents   = (table_t*) apsp->first;    
    table_t* distances = (table_t*) apsp->second;

    free(apsp);

    // Fill dp table

    c = clock();
    fill_steiner_dp_table_cpu(costs, g, terminals, distances);
    printf("\tDP Table fill: %f\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    // Free table

    free_table(distances);
    free_table(parents);

    return costs;

}
