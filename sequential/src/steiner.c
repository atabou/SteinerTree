
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

    int** costs = (int**) malloc(sizeof(int*) * V);
    graph*** trees = (graph***) malloc(sizeof(graph**) * V);
    
    for(int v=0; v < V; v++) {

        costs[v] = (int*) malloc(sizeof(int) * P);
        trees[v] = (graph**) malloc(sizeof(graph*) * P);
        
    }

    // All pairs shortest path

    pair* apsp = all_pairs_shortest_path(g);

    graph*** paths  = (graph***) apsp->first;
    int** distances =    (int**) apsp->second;

    free(apsp);

    // Fill base cases

    for(int v=0; v < V; v++) {

        long long mask = 1ll << (T - 1);

        for(int pos=0; pos < T; pos++) {

            int u = g->reverse_hash[get_element(terminals, pos)];

            costs[v][mask - 1] = distances[v][u];
            trees[v][mask - 1] = paths[v][u];

            mask = mask >> 1;

        }

    }

    for(int k=2; k <= T; k++) {

        long long mask = 0;

        while( next_combination(T, k, &mask) ) { // while loop runs T choose k times (nCr).

            set_t* X = get_subset(terminals, mask);

            for(int v=0; v < V; v++) { // O(T * 2^T * V * (V + E))

                set_t* W = bfs(g, g->hash[v], steiner_verification, X); // O(V+E)
            
                int min = INT_MAX;
                graph* min_tree = NULL;

                for(int i=0; i < set_size(W); i++) { // O(T * 2^T * (V+E))

                    int w = g->reverse_hash[get_element(W, i)];

                    if( element_exists(g->hash[w], X) ) { // O(V + E)

                        long long submask = 1ll << (T - find_position(terminals, g->hash[w]) - 1);

                        int cost = distances[v][w] + costs[w][(mask & ~submask) - 1];

                        if(cost < min) {
                            min = cost;
                            min_tree = graph_union(paths[v][w], trees[w][(mask & ~submask) - 1]); // (V + E)
                        }

                    } else { // O(2^T (V+E))

                        for(long long submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                            int cost = distances[v][w] + costs[w][submask - 1] + costs[w][(mask & ~submask) - 1];

                            if(cost < min) {

                                min = cost;

                                destroy_graph(min_tree);

                                graph* tmp_tree = graph_union(trees[w][submask - 1], trees[w][(mask & ~submask) - 1]); // O(V + E)
                                min_tree = graph_union(paths[v][w], tmp_tree);
                                destroy_graph(tmp_tree);

                            }

                        }

                    }

                }

                costs[v][mask - 1] = min;
                trees[v][mask - 1] = min_tree;

                destroy_set(W);

            }

            destroy_set(X);

        }

    }

    // print_table(costs, V, P);

    // Extract minimum from table.

    int min_index = 0;
    
    for(int i=1; i < V; i++) {
        
        if(costs[i][P - 1] < costs[min_index][P - 1]) {
            min_index = i;
        }
    
    }

    graph* tree = trees[min_index][P - 1];
    int    cost = costs[min_index][P - 1];

    // Free table

    free_table(costs, V, P);
    free_table(distances, V, V);

    for(int i=0; i < V; i++) {

        for(long long j=0; j<P; j++) {

            if(i != min_index || j != P - 1) {
                destroy_graph(trees[i][j]);
            }

            trees[i][j] = NULL;

        }

        free(trees[i]);
        trees[i] = NULL;

    }

    free(trees);
    trees = NULL;

    return make_pair(tree, cost);

}