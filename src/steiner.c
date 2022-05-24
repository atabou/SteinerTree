
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#include "steiner.h"
#include "steiner.cuda.h"

#include "shortestpath.h"

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
                    
                    uint32_t min = INT32_MAX;
                    
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

float steiner_tree(graph_t* g, set_t* terminals, table_t* distances) {

    // Initialize DP table.
    
    table_t* costs = make_table(g->vrt, (int32_t) pow(2, terminals->size) - 1);

    // Fill dp table.

    fill_steiner_dp_table_cpu(costs, g, terminals, distances);

    // Calculate minimum from table.

    float min = FLT_MAX;

    for(int32_t i=costs->m - 1; i < costs->m * distances->n; i+=costs->m) {
        
        if(costs->vals[i] < min) {
            min = costs->vals[i];
        }

    }

    // Free table
    
    /* print_table(costs); */
   
    free(costs);

    return min;

}
