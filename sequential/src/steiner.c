
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "pair.h"
#include "graph.h"
#include "steiner.h"
#include "set.h"

int verify(graph* g, int v, void* input) {

    int check1 = degree(g, v) >= 3;
    int check2 = element_exists(v, (set_t*) input);

    return check1 || check2;

}

pair* steiner_bottom_up(graph* g, set_t* terminals) {

    int**    costs =    (int**) malloc(sizeof(int)    * 11);
    graph*** trees = (graph***) malloc(sizeof(graph**) * 11);

    long long n = (long long) pow(2, set_size(terminals));

    for(int v=1; v<11; v++) {

        costs[v-1] = (int*) malloc(sizeof(int) * n);
        trees[v-1] = (graph**) malloc(sizeof(graph*) * n);

    }

    // All pairs shortest path

    // Fill array with corresponding sortest paths

    // Start building the array in order by incrementing a mask until 2^|T|

}

pair* streiner_tree_dp(graph* g, set_t* terminals, int v) {

    if(set_size(terminals) == 1) {

        return shortest_path(g, get_element(terminals, 0), v);

    } else {

        int w = dfs(g, v, verify, terminals);

        pair*  sp_pair = shortest_path(g, v, w);

        graph* sp_path = (graph*) sp_pair->first;
        int    sp_dist =    (int) sp_pair->second;

        free(sp_pair);

        graph* min_tree = NULL;
        int    min_cost = INT_MAX;

        if( element_exists(w, terminals) ) {

            set_t* X = remove_element(w, terminals);

            pair*  dp_pair = streiner_tree_dp(g, X, w);

            destroy_set(X);

            graph* dp_tree = (graph*) dp_pair->first;
            int    dp_cost =    (int) dp_pair->second;

            free(dp_pair);

            min_tree = graph_union(sp_path, dp_tree);
            min_cost = sp_dist + dp_cost;

            destroy_graph(dp_tree);

        } else {

            for(long long mask=1; mask<pow(2, set_size(terminals)) - 1; mask++) {

                set_t* X = get_subset(terminals, mask); // Constructs the subset X from the given mask.

                pair*  dp_pair1 = streiner_tree_dp(g, X, w);

                destroy_set(X);

                graph* dp_tree1 = (graph*) dp_pair1->first;
                int    dp_cost1 =    (int) dp_pair1->second;

                free(dp_pair1);

                set_t* Y = get_subset(terminals, (~mask) & ((long long) pow(2, set_size(terminals)) - 1)); // Constructs the subset terminals - X

                pair*  dp_pair2 = streiner_tree_dp(g, Y, w);

                destroy_set(Y);

                graph* dp_tree2 = (graph*) dp_pair2->first;
                int    dp_cost2 =    (int) dp_pair2->second;

                free(dp_pair2);

                if(dp_cost1 + dp_cost2 + sp_dist < min_cost) {
                    
                    destroy_graph(min_tree);

                    graph* tmp_tree = graph_union(dp_tree1, dp_tree2);

                    int    min_cost = dp_cost1 + dp_cost2;

                    min_tree = graph_union(tmp_tree, sp_path);
                    min_cost = min_cost + sp_dist;

                    destroy_graph(tmp_tree);

                }

                destroy_graph(dp_tree1);
                destroy_graph(dp_tree2);

            }
            
        }

        destroy_graph(sp_path);

        return make_pair(min_tree, (void*) min_cost);

    }

}

graph* steiner_tree(graph* g, set_t* terminals) {

    graph* min_tree = NULL;
    int min_cost = INT_MAX;

    to_graphviz(g, "test.dot");

    for(int i=1; i<11; i++) {

        pair* p = streiner_tree_dp(g, terminals, i);

        graph* steiner = (graph*) p->first;
        int    cost = (int) p->second;

        if(cost < min_cost) {

            destroy_graph(min_tree);

            min_tree = steiner;
            min_cost = cost;

        }

    }

    return min_tree;

}