
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "steiner.h"
#include "bfs.h"
#include "common.h"
#include "shortestpath.h"

int steiner_verification(graph* g, int v, void* input) {

    int check1 = degree(g, v) >= 3;
    int check2 = element_exists(v, (set_t*) input);

    return check1 || check2;

}

pair* steiner_tree(graph* g, set_t* terminals) {

    int V = g->nVertices;
    int T = set_size(terminals);
    long long P =  (long long) pow(2, T) - 1;

    int** costs  =  (int**) malloc(sizeof(int*) * V);
    
    for(int v=0; v < V; v++) {

        costs[v] = (int*) malloc(sizeof(int) * P);
        
    }

    // All pairs shortest path

    pair* apsp = all_pairs_shortest_path(g);
    
    int** distances = (int**) apsp->second;

    free(apsp);

    // Fill tables

    for(int k=1; k <= T; k++) {

        long long mask = 0;

        while( next_combination(T, k, &mask) ) { // while loop runs T choose k times (nCr).

            set_t* X = get_subset(terminals, mask);

            for(int v=0; v < V; v++) { // 

                if(k == 1) {

                    int u = g->reverse_hash[get_element(X, 0)];
                    costs[v][mask - 1] = distances[v][u];
                    
                } else {

                    set_t* W = bfs(g, g->hash[v], steiner_verification, X); // O(V+E)
            
                    int min = INT_MAX;
                    
                    for(int i=0; i < set_size(W); i++) { // O(T * 2^T * (V+E))

                        int w = g->reverse_hash[get_element(W, i)];

                        if( element_exists(g->hash[w], X) ) { // O(V + E)

                            long long submask = 1ll << (T - find_position(terminals, g->hash[w]) - 1);

                            int cost = distances[v][w] + costs[w][(mask & ~submask) - 1];

                            if(cost < min) {

                                min = cost;

                            }

                        } else { // O(2^T (V+E))

                            for(long long submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                                int cost = distances[v][w] + costs[w][submask - 1] + costs[w][(mask & ~submask) - 1];

                                if(cost < min) {

                                    min = cost;

                                }

                            }

                        }

                    }

                    costs[v][mask - 1] = min;

                    destroy_set(W);

                }
                
            }

            destroy_set(X);

        }

    }

    // Extract minimum from table.

    int min_index = 0;
    
    for(int i=1; i < V; i++) {
        
        if(costs[i][P - 1] < costs[min_index][P - 1]) {
            min_index = i;
        }
    
    }

    int    cost = costs[min_index][P - 1];

    // Free table

    free_table(costs, V, P);
    free_table(distances, V, V);

    return make_pair(NULL, cost);

}