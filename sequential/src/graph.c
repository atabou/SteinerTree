
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"
#include "common.h"

graph* make_graph(int max_id) {

    graph* g = (graph*) malloc(sizeof(graph));

    g->capacity     = 0;
    g->nVertices    = 0;
    g->max_id       = max_id + 1;

    g->hash         = NULL;
    g->deg          = NULL;
    g->lst          = NULL;

    g->reverse_hash = (int*) malloc(sizeof(graph) * (max_id + 1));

    for(int i=0; i<max_id + 1; i++) {
        g->reverse_hash[i] = -1;
    }    

    return g;

}


int number_of_vertices(graph* g) {

    return g->nVertices;

}

int number_of_edges(graph* g) {

    int n = 0;

    for(int i=0; i<number_of_vertices(g); i++) {
        n += g->deg[i];
    }

    return n;

}

int degree(graph* g, int id) {

    if(id >= 0 && id < g->max_id) {

        int k = g->reverse_hash[id];

        if(k != -1) {

            return g->deg[k];

        } else {

            return -1;

        }

    }

}


void insert_vertex(graph* g, int id) {

    if(id >= 0 && id < g->max_id && g->reverse_hash[id] == -1) {

        if(g->capacity == 0) {

            g->hash    =    (int*) malloc( sizeof(int) );
            g->deg     =    (int*) malloc( sizeof(int) );
            g->lst     = (llist**) malloc(sizeof(llist*));
            
            g->capacity = 1;

        } else if(g->nVertices >= g->capacity) { // O(V) normally, O(1) ammortized.

            g->hash    =    (int*) realloc(g->hash, sizeof(int) * 2 * g->capacity);
            g->deg     =    (int*) realloc(g->deg, sizeof(int) * 2 * g->capacity);
            g->lst     = (llist**) realloc(g->lst, sizeof(llist*) * 2 * g->capacity);

            g->capacity = 2 * g->capacity;

        }

        g->hash[g->nVertices] = id;
        g->reverse_hash[id]   = g->nVertices;
        g->deg[g->nVertices]  = 0;
        g->lst[g->nVertices]  = NULL;

        g->nVertices = g->nVertices + 1;

    }

}

void insert_edge(graph* g, int id1, int id2, int w) {

    if(id1 >= 0 && id2 >= 0 && id1 < g->max_id && id2 < g->max_id) {

        int internal1 = g->reverse_hash[id1];
        int internal2 = g->reverse_hash[id2];

        if(internal1 != -1 && internal2 != -1) {

            llist* e = g->lst[internal1];

            while(e != NULL) {

                if(e->data == internal2) {
                    return;
                }

                e = e->next;

            }

            g->lst[internal1] = llist_add(g->lst[internal1], internal2, w);
            g->deg[internal1]++;

        }

    }

}

void remove_vertex(graph* g, int id) {

    if(id > 0 && id < g->max_id) {

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

    if(id1 >= 0 && id2 >= 0 && id1 < g->max_id && id2 < g->max_id) {

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

                insert_edge(g, i, used[j], 1);

            }

        }

    }

    return g;

}

// O(max_id + V + E)
graph* graph_union(graph* g1, graph* g2) {

    graph* g = make_graph( max(g1->max_id, g2->max_id) );

    for(int i=0; i<g1->nVertices; i++) {
        insert_vertex(g, g1->hash[i]);
    }

    for(int i=0; i<g1->nVertices; i++) {

        llist* e = g1->lst[i];
        
        while(e != NULL) {

            insert_edge(g, g1->hash[i], g1->hash[e->data], e->weight);
            e = e->next;

        }

    }

    for(int j=0; j<g2->nVertices; j++) {
        insert_vertex(g, g2->hash[j]);
    }

    for(int i=0; i<g2->nVertices; i++) {

        llist* e = g2->lst[i];
        
        while(e != NULL) {

            insert_edge(g, g2->hash[i], g2->hash[e->data], e->weight);
            e = e->next;
            
        }

    }

    return g;

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

    if(g == NULL) {
        return;
    }

    for(int i=0; i<g->nVertices; i++) {

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


