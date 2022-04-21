
#include <stdlib.h>
#include <limits.h>

#include "heap.h"
#include "shortestpath.h"

void dijkstra(graph* g, int internal1, int* distances, int* parents, int start) {

    // Initialize Single Source Shortest Path

    for(int i=0; i<g->nVertices; i++) {

        distances[i + start] = INT_MAX;
        parents[i + start] = -1;

    }

    distances[internal1 + start] = 0;

    // Initialize minheap.

    heap* pq = make_heap(FIBONACCI, g->nVertices);

    for(int i=0; i < g->nVertices; i++) {
        pq->insert(pq, i, distances[i + start]);
    }

    // Calculates shortest path

    while( !pq->empty(pq) ) {

        int u = pq->extract_min(pq);

        llist* curr = g->lst[u];

        while(curr != NULL) {

            pair* p = (pair*) curr->data;

            int v = (int) p->first;
            int w = (int) p->second;


            if(distances[v + start] > distances[u + start] + w) {

                distances[v + start] = distances[u + start] + w;
                pq->decrease_key(pq, v, distances[v + start]);
                parents[v + start] = u;

            }

            curr = curr->next;

        }

    }

    pq->destroy(pq); // O(1) because the priority q is empty at this point.

}

pair* all_pairs_shortest_path(graph* g) {

    int* distances = (int*) malloc(sizeof(int) * g->nVertices * g->nVertices);
    int* parents = (int*) malloc(sizeof(int) * g->nVertices * g->nVertices);
    
    for(int v=0; v<g->nVertices; v++) {

        dijkstra(g, v, distances, parents, v * g->nVertices); // (E + V log (V))

    }

    // return make_pair(paths, distances);

    return make_pair(parents, distances);

}