
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "llist.h"
#include "graph.h"
#include "heap.h"
#include "pair.h"

struct graph {

    /**
    * Hash that links the internal ID of a vertex to a user specified ID.
    * If next_slot is less than nVertices then the slot represented by next_slot represents an empty "parent" slot.
    * 
    */
    int*    hash;

    int     nVertices; // The number of vertices in the graph. Also represents the leftmost empty position.
    int     capacity; // The capacity of the hash table.

    int*    reverse_hash; // Links the user specified IDs to its corresponding internal ID.
    int     max_id; // The biggest user specified ID. Also the size of the reverse hash table minus - 1.
    

    int*    deg; // Represents the degree of the node. If the degree of the node is -1 then the node does not exits.
    llist** lst; // The adjacency list of the graph.

};


graph* make_graph(int max_id) {

    graph* g = (graph*) malloc(sizeof(graph));

    g->capacity     = 0;
    g->nVertices    = 0;
    g->max_id       = max_id;

    g->hash         = NULL;
    g->deg          = NULL;
    g->lst          = NULL;

    g->reverse_hash = (int*) malloc(sizeof(graph) * (max_id + 1));

    for(int i=0; i<max_id + 1; i++) {
        g->reverse_hash[i] = -1;
    }    

    return g;

}

void insert_vertex(graph* g, int id) {

    if(g->capacity == 0) {

        g->hash = (int*) malloc( sizeof(int) );
        g->deg  = (int*) malloc( sizeof(int) );
        g->lst  = (llist**) malloc(sizeof(llist*));
        
        g->capacity = 1;

    }

    if(g->nVertices >= g->capacity) { // O(V) normally, O(1) ammortized.

        g->hash = (int*) realloc(g->hash, sizeof(int) * 2 * g->capacity);
        g->deg  = (int*) realloc(g->deg, sizeof(int) * 2 * g->capacity);
        g->lst  = (llist**) realloc(g->lst, sizeof(llist*) * 2 * g->capacity);

        g->capacity = 2 * g->capacity;

    }

    g->hash[g->nVertices] = id;
    g->reverse_hash[id]   = g->nVertices;
    g->deg[g->nVertices]  = 0;
    g->lst[g->nVertices]  = NULL;

    g->nVertices = g->nVertices + 1;

}

void insert_edge(graph* g, int id1, int id2, int w) {

    if(id1 >= 0 && id2 >= 0 && id1 <= g->max_id && id2 <= g->max_id) {

        int internal1 = g->reverse_hash[id1];
        int internal2 = g->reverse_hash[id2];

        if(internal1 != -1 && internal2 != -1) {

            g->lst[internal1] = llist_add(g->lst[internal1], internal2, w);
            g->deg[internal1]++;

        }


    }

}

void remove_vertex(graph* g, int id) {

    if(id > 0 && id <g->max_id) {

        int internal = g->reverse_hash[id];

        if(internal != -1) {

            // Removes all edges related to id.

            for(int i=0; i<g->nVertices; i++) { // O(E)

                if(i == internal) {

                    destroy_llist(g->lst[i]);
                    g->lst[i] = NULL;
                    g->deg[i] = 0;

                } else {

                    remove_edge(g, i, internal);

                }

            }

            // Move all upstream vertices and their ids down.

            for(int i=internal; i<g->nVertices - 1; i++) { // O(V)

                g->reverse_hash[g->hash[i+1]] = i;
                g->hash[i] = g->hash[i+1];
                g->deg[i] = g->deg[i+1];
                g->lst[i] = g->lst[i+1];

            }

            // Clear the corresponding reverse hash

            g->reverse_hash[id] = -1;

            // Set the last position in the hash as availble
            
            g->hash[g->nVertices - 1] = -1;
            g->deg[g->nVertices - 1]  = 0;
            g->lst[g->nVertices - 1]  = NULL;
            g->nVertices--;
            
        }

    }

}

void remove_edge(graph* g, int id1, int id2) {

    if(id1 >= 0 && id2 >= 0 && id1 <= g->max_id && id2 <= g->max_id) {

        int internal1 = g->reverse_hash[id1];
        int internal2 = g->reverse_hash[id2];

        if(g->lst[internal1] != NULL) {

            if(g->lst[internal1]->data == internal2) {

                llist* del = g->lst[internal1];
                g->lst[internal1] = g->lst[internal1]->next;
                del->next = NULL;

                destroy_llist(del);
                del = NULL;

                g->deg[internal1]--;

            } else {

                llist* curr = g->lst[internal1];

                while(curr->next != NULL && curr->next->data != internal2) {
                    curr = curr->next;
                }

                if(curr->next != NULL) {

                    llist* del = curr->next;
                    curr->next = curr->next->next;
                    del->next = NULL;
                    
                    destroy_llist(del);
                    del = NULL;

                    g->deg[internal1]--;

                }

            }

        }

    }


}

graph* make_randomly_connected_graph(int max_id) {

    graph* g = make_graph(max_id);

    for(int i=0; i<g->max_id + 1; i++) {
        insert_vertex(g, i);
    }

    int used[max_id + 1];

    for(int i=0; i<max_id+1; i++) {

        for(int j=0; j<max_id + 1; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<max_id + 1; j++) { // Shuffle array

            int x = rand() % (max_id + 1);
            int y = rand() % (max_id + 1);

            int tmp = used[x];
            used[x] = used[y];
            used[y] = tmp;

        }

        int deg = rand() % max_id;

        for(int j=0; j<deg; j++) {

            if(used[j] != -1) {

                printf("(%d, %d)\n", i, used[j]);
                insert_edge(g, i, used[j], 1);

            }

        }

    }

    return g;

}

pair* shortest_path(graph* g, int v1, int v2) {

    // Check boundary conditions

    if(v1 < 0 || v2 < 0 || v1 >= g->nVertices || v2 >= g->nVertices) {
        return NULL;
    }

    int internal1 = g->reverse_hash[v1];
    int internal2 = g->reverse_hash[v2];

    // Initialize single source shortest path

    int distances[g->nVertices];
    int parents[g->nVertices];

    for(int i=0; i<g->nVertices; i++) {

        distances[i] = INT_MAX;
        parents[i] = -1;

    }

    distances[v1] = 0;

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

    graph* path = make_graph(g->max_id);

    insert_vertex(path, v2);

    int child  = v2;
    int vertex = parents[v2];

    while(vertex != -1) {

        insert_vertex(path, vertex);

        insert_edge(g, vertex, child, distances[child] - distances[vertex]);
        insert_edge(g, child, vertex, distances[child] - distances[vertex]);
        
        child = vertex;
        vertex = parents[vertex];

    }

    pair* result = (pair*) malloc(sizeof(pair));

    result->first = (void*) path;
    result->second = (void*) distances[v2];

    return result;

}

void to_graphviz(graph* g, char* filename) {

    FILE* fp = fopen(filename, "w");

    fprintf(fp, "digraph G {\n\n");

    for(int i=0; i<g->nVertices; i++) {

        fprintf(fp, "\t%d [label=\"%d\"];\n", i, g->hash[i]);

    }

    fprintf(fp, "\n");

    for(int i=0; i<g->nVertices; i++) {

        llist* curr = g->lst[i];

        while(curr != NULL) {

            fprintf(fp, "\t%d -> %d [label=\"%d\"];\n", i, curr->data, curr->weight);  
            curr = curr->next;

        }

    }

    fprintf(fp, "\n}");

    fclose(fp);

}

void destroy_graph(graph* g) {

    for(int i=0; i<g->capacity; i++) {

        destroy_llist(g->lst[i]);
        g->lst[i] = NULL;

    }

    g->capacity = 0;
    g->max_id = 0;
    g->nVertices = 0;

    free(g->hash);
    g->hash = NULL;

    free(g->deg);
    g->deg = NULL;

    free(g->lst);
    g->lst = NULL;

    free(g->reverse_hash);
    g->reverse_hash = NULL;

    free(g);

}