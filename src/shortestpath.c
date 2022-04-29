
#include <stdlib.h>
#include <limits.h>

#include "shortestpath.h"
#include "heap.h"
#include "table.h"

void dijkstra(graph_t* g, uint32_t src, table_t* distances, table_t* parents, uint32_t start) {

    // Initialize Single Source Shortest Path

    for(uint32_t i=0; i<g->vrt; i++) {

        distances->vals[i + start] = UINT32_MAX;
        parents->vals[i + start] = UINT32_MAX;

    }

    distances->vals[src + start] = 0;

    // Initialize minheap.

    heap_t* pq = make_heap(FIBONACCI, g->vrt);

    for(uint32_t i=0; i < g->vrt; i++) {
        pq->insert(pq, i, distances->vals[i + start]);
    }

    // Calculates shortest path

    while( !pq->empty(pq) ) {

        uint32_t u = pq->extract_min(pq);

        llist_t* edges = g->lst[u];

        while(edges != NULL) {

            uint32_t v = edges->dest;
            uint32_t w = edges->weight;
           
            if(distances->vals[v + start] > distances->vals[u + start] + w) {

                distances->vals[v + start] = distances->vals[u + start] + w;
                pq->decrease_key(pq, v, distances->vals[v + start]);
                parents->vals[v + start] = u;

            }

            edges = edges->next;

        }

    }

    pq->destroy(pq); // O(1) because the priority q is empty at this point.

}

pair_t* all_pairs_shortest_path(graph_t* g) {

    // Initialize tables

    table_t* distances = make_table(g->vrt, g->vrt);
    table_t* parents = make_table(g->vrt, g->vrt);

    // Calculate shortest path from every source node.

    for(uint32_t v=0; v<g->vrt; v++) {

        dijkstra(g, v, distances, parents, v * g->vrt); // (E + V log (V))

    }

    return make_pair(parents, distances);

}
