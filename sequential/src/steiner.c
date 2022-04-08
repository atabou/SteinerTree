
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "pair.h"
#include "graph.h"
#include "steiner.h"
#include "set.h"

int verify(graph* g, int v, void* input) {

    pair* p = (pair*) input;

    int check1 = degree(g, v) >= 3;

    int  n = (int) ((pair*) p)->second;
    int* t = (int*) ((pair*) p)->first;

    int check2 = 0;
    for(int i=0; i<n; i++) {
        check2 = check2 || t[i] == v;
    }

    return check1 && check2;

}


pair* streiner_tree_dp(graph* g, set_t* terminals, int v) {

    if(set_size(terminals) == 1) {

        return shortest_path(g, get_element(terminals, 0), v);

    } else {

        pair* dfs_input = make_pair(terminals, set_size(terminals));
        
        int w = dfs(g, v, verify, dfs_input);
        
        free(dfs_input);

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

                set_t* Y = get_subset(terminals, (~mask) & (long long) pow(2, set_size(terminals))); // Constructs the subset terminals - X

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

                    free(tmp_tree);

                }

                destroy_graph(dp_tree1);
                destroy_graph(dp_tree2);

            }
            
        }

        destroy_graph(sp_path);

        return make_pair(min_tree, min_cost);

    }

}

graph* steiner_tree(graph* g, int* terminals, int n) {

    // for(int i=0; i<n; i++) {

    //     pair* steiner = streiner_tree_dp(g, terminals, n, i+1);

    //     graph* g = (pair* )

    // }

}