
#include <stdlib.h>
#include <limits.h>

#include "graph.h"
#include "heap.h"

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

    heap* pq = make_heap(FIBONACCI, g->V);

    for(int i=0; i < g->V; i++) {
        pq->insert(pq, i, distances[i]);
    }

    // Calculates shortest path

    while( !pq->empty(pq) ) {

        int u = pq->extract_min(pq);

        llist* curr = g->lst[u];

        while(curr != NULL) {

            int v = curr->data;

            if(distances[v] > distances[u] + curr->weight) {

                distances[v] = distances[u] + curr->weight;
                pq->decrease_key(pq, v, distances[v]);
                parents[v] = u;

            }

            curr = curr->next;

        }

    }

    pq->destroy(pq); // O(1) because the priority q is empty at this point.

}