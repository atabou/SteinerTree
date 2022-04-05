
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "llist.h"
#include "graph.h"
#include "heap.h"

struct graph {

    int     V;
    void**  data;
    int*    deg;
    llist** lst;

};

/**
 * @brief Initializes a graph with V vertex.
 * 
 * @param V 
 * @param data 
 * @return graph* 
 */
graph* make_graph(int V) {

    graph* g = (graph*) malloc(sizeof(graph));

    g->V = V;

    g->deg = (int*) malloc(sizeof(int) * g->V);

    for(int i=0; i<g->V; i++) {
        g->deg[i] = 0;
    }

    g->lst = (llist**) malloc(sizeof(llist*) * g->V);

    for(int i=0; i<g->V; i++) {
        g->lst[i] = NULL;
    }

    return g;

}

/**
 * @brief Randomly connects all nodes randomly.
 * 
 * @param g an unconnected graph.
 */
graph* make_randomly_connected_graph(int V) {

    graph* g = make_graph(V);

    for(int i=0; i<g->V; i++) {
        g->deg[i] = rand() % V;
    }

    int used[g->V];

    for(int i=0; i<V; i++) {

        for(int j=0; j<V; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<g->V; j++) {

            int x = rand()%V;
            int y = rand()%V;

            int tmp = used[x];
            used[x] = used[y];
            used[y] = used[x];

        }

        for(int j=0; j<g->deg[i]; j++) {

            if(used[j] != -1) {

                g->lst[i] = llist_add(g->lst[i], used[j], 1);

            }

        }

    }

    return g;

}

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

void to_graphviz(graph* g, char* filename) {

    FILE* fp = fopen(filename, "w");

    for(int i=0; i<g->V; i++) {

        fprintf(fp, "%d [label=\"%d\"];\n", g->data[i], g->data[i]);

    }

    for(int i=0; i<g->V; i++) {

        if(g->data[i] != NULL) {

            llist* curr = g->lst[i];

            while(curr != NULL) {

                fprintf(fp, "%d -> %d [label=\"d\"]", i, curr->data, curr->weight);                
                curr = curr->next;

            }

        }

    }

    fclose(fp);

}

void destroy_graph(graph* g) {

    for(int i=0; i<g->V; i++) {

        destroy_llist(g->lst[i]);
        g->lst[i] = NULL;

    }

    free(g->deg);

}