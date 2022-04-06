
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "llist.h"
#include "graph.h"
#include "set.h"


graph* steiner_tree(graph* g, int* terminals, int n) {

    int  t = terminals[0];
    
    int* ts = (int*) malloc(sizeof(int) * (n - 1));
    
    for(int i=0; i<n-1; i++) {
        ts[i] = terminals[i+1];
    }

    int** pset = powerset(ts, n - 1);

    free(ts);
    free(pset);

}

int main(int argc, char** argv) {

    int max_id = 10;
    
    graph* g = make_randomly_connected_graph(max_id);

    to_graphviz(g, "test.dot");
    shortest_path(g, 1, 2);

    destroy_graph(g);
    
    return 0;

}
