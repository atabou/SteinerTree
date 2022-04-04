
#include <limits.h>

#include "graph.h"

int shortest_path(graph* g, int v1, int v2) {

    // Check boundary conditions

    if(v1 < 0 && v2 < 0 && v1 >= g->V && v2 >= g->V) {
        return -1;
    }

    // Initialize single source shortest path

    int distances[g->V];
    int parents[g->V];

    for(int i=0; i<g->V; i++) {

        distances[i] = INT_MAX;
        parents[i] = -1;

    }

    distances[v1] = 0;

    // Initialize minheap.


}