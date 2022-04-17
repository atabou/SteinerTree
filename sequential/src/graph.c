
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

    return -1;

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

                int comp = (int) ((pair*) e->data)->first;

                if(comp == internal2) {
                    return;
                }

                e = e->next;

            }

            g->lst[internal1] = llist_add(g->lst[internal1], make_pair((void*) internal2, (void*) w));
            g->deg[internal1]++;

        }

    }

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

            pair* p = (pair*) e->data;

            int dest = g1->hash[(int) p->first];
            int w    = (int) p->second;

            insert_edge(g, g1->hash[i], dest, w);
            e = e->next;

        }

    }

    for(int j=0; j<g2->nVertices; j++) {
        insert_vertex(g, g2->hash[j]);
    }

    for(int i=0; i<g2->nVertices; i++) {

        llist* e = g2->lst[i];
        
        while(e != NULL) {

            pair* p = (pair*) e->data;

            int dest = g2->hash[(int) p->first];
            int w    = (int) p->second;

            insert_edge(g, g2->hash[i], dest, w);
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

            pair* p = (pair*) curr->data;

            int dest = (int) p->first;
            int w    = (int) p->second;

            fprintf(fp, "\t%d -> %d [label=\"%d\"];\n", i, dest, w);  
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

        destroy_llist(g->lst[i], free);
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


