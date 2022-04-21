
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "steiner.h"
#include "common.h"
#include "shortestpath.h"

pair* steiner_tree(graph* g, set_t* terminals) {

    int V = g->nVertices;
    int T = set_size(terminals);
    long long P =  (long long) pow(2, T) - 1;

    int* costs = (int*) malloc(sizeof(int) * V * P);

    printf(" - V: %d, T: %d, time: ", V, T);
    fflush(stdout);

    // All pairs shortest path

    pair* apsp = all_pairs_shortest_path(g);
    
    int* distances = (int*) apsp->second;

    free(apsp);

    // Fill tables

    clock_t c = clock();

    for(int k=1; k <= T; k++) {

        long long mask = 0;

        while( next_combination(T, k, &mask) ) { // while loop runs T choose k times (nCr).

            set_t* X = get_subset(terminals, mask);

            for(int v=0; v < V; v++) {

                if(k == 1) {

                    int u = g->reverse_hash[get_element(X, 0)];
                    costs[v*P + (mask - 1)] = distances[v * V + u];
                    
                } else {
                    
                    int min = INT_MAX;
                    
                    for(int w=0; w < V; w++) { // O(T * 2^T * (V+E))

                        if( element_exists(g->hash[w], X) ) { // O(V + E)

                            long long submask = 1ll << (T - find_position(terminals, g->hash[w]) - 1);

                            int cost = distances[v * V + w] + costs[w * P + ((mask & ~submask) - 1)];

                            if(cost < min) {

                                min = cost;

                            }

                        } else if(degree(g, g->hash[w]) >= 3) { // O(2^T (V+E))

                            for(long long submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                                int cost = distances[v * V + w] + costs[w * P + (submask - 1)] + costs[w * P + ((mask & ~submask) - 1)];

                                if(cost < min) {

                                    min = cost;

                                }

                            }

                        }

                    }

                    costs[v * P + (mask - 1)] = min;

                }
                
            }

            destroy_set(X);

        }

    }

    printf("%f\n", (double) (clock() - c) / CLOCKS_PER_SEC );

    // Extract minimum from table.

    int min_index = 0;
    
    for(int i=1; i < V; i++) {

        if(costs[i * P + (P - 1)] < costs[min_index * P + (P - 1)]) {
            min_index = i;
        }
    
    }

    int cost = costs[min_index * P + (P - 1)];

    // Free table

    free(costs);
    free(distances);

    return make_pair(NULL, cost);

}