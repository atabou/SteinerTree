
#include <stdlib.h>
#include <limits.h>

#include "heap.h"
#include "shortestpath.h"

pair* dijkstra(graph* g, int internal1) {

    // Initialize Single Source Shortest Path

    int* distances = (int*) malloc(sizeof(int) * g->nVertices);
    int* parents   = (int*) malloc(sizeof(int) * g->nVertices);

    for(int i=0; i<g->nVertices; i++) {

        distances[i] = INT_MAX;
        parents[i] = -1;

    }

    distances[internal1] = 0;

    // Initialize minheap.

    heap* pq = make_heap(FIBONACCI, g->nVertices);

    for(int i=0; i < g->nVertices; i++) {
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

    return make_pair(distances, parents);

}

// O(E)
graph* construct_path(graph* g, int* distances, int* parents, int target) {

    graph* path = make_graph(g->max_id);

    insert_vertex(path, g->hash[target]);

    int child  = target;
    int vertex = parents[target];

    while(vertex != -1) {

        insert_vertex(path, g->hash[vertex]);

        insert_edge(path, g->hash[vertex], g->hash[child], distances[child] - distances[vertex]);
        insert_edge(path, g->hash[child], g->hash[vertex], distances[child] - distances[vertex]);

        child = vertex;
        vertex = parents[vertex];

    }

    return path;

}

pair* shortest_path(graph* g, int v1, int v2) {

    // Check boundary conditions

    if(v1 < 0 || v2 < 0 || v1 >= g->max_id || v2 >= g->max_id) {
        return NULL;
    }

    int internal1 = g->reverse_hash[v1];
    int internal2 = g->reverse_hash[v2];

    if(internal1 == -1 || internal2 == -1) {
        return NULL;
    }

    pair* p = dijkstra(g, internal1);

    int* distances = (int*) p->first;
    int* parents = (int*) p->second;

    graph* path = construct_path(g, distances, parents, internal2);
    int distance = distances[internal2];

    free(distances);
    free(parents);

    return make_pair((void*) path, (void*) distance);

}

pair* all_pairs_shortest_path(graph* g) {

    int** distances =    (int**) malloc(sizeof(int*) * g->nVertices);
    graph*** paths  = (graph***) malloc(sizeof(graph**) * g->nVertices);

    for(int v=0; v<g->nVertices; v++) {

        pair* sp = dijkstra(g, v); // (E + V log (V))

        // Construct distances

        distances[v] = (int*) sp->first;
        
        // Construct paths

        paths[v] = (graph**) malloc(sizeof(graph*) * g->nVertices);
        
        int* parents = (int*) sp->second;

        for(int u=0; u<g->nVertices; u++) { // O(V*E)

            paths[v][u] = construct_path(g, distances[v], parents, u); // O(E)

        }

        free(parents);
        free(sp);

    }

    return make_pair(paths, distances);

}