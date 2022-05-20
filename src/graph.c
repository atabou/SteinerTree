
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"


graph_t* make_graph() {

    graph_t* g = (graph_t*) malloc(sizeof(graph_t));

    g->max = 0;
    g->vrt = 0;
    
    g->deg = NULL;
    g->lst = NULL;    

    return g;

}

uint32_t insert_vertex(graph_t* g) {

    if(g->vrt < INT32_MAX) {

        if(g->max == 0) {

            g->deg = (int32_t*) malloc(sizeof(int32_t));
            g->lst = (llist_t**) malloc(sizeof(llist_t*));

            g->max = 1;

        } else if(g->max * 2 < g->max) { // Executes if there is an overflow

            g->deg = (int32_t*) realloc(g->deg, sizeof(int32_t) * INT32_MAX);
            g->lst = (llist_t**) realloc(g->lst, sizeof(llist_t*) * INT32_MAX);

            g->max = INT32_MAX;

        } else if(g->vrt >= g->max) {

            g->deg = (int32_t*) realloc(g->deg, sizeof(int32_t) * 2 * g->max);
            g->lst = (llist_t**) realloc(g->lst, sizeof(llist_t*) * 2 * g->max);

            g->max = 2 * g->max;

        }

        g->deg[g->vrt]  = 0;
        g->lst[g->vrt]  = NULL;

        g->vrt = g->vrt + 1;

    }

    return g->vrt - 1;

}

void insert_edge(graph_t* g, int32_t src, int32_t dest, float w) {

    if(src < g->vrt && dest < g->vrt) {

        llist_t* e = g->lst[src];

        while (e != NULL) {

            if(e->dest == dest) {
                return;
            }

            e = e->next;


        }

        g->lst[src] = llist_add(g->lst[src], dest, w);
        g->deg[src]++;
        
    }

}


void to_graphviz(graph_t* g, char* filename) {

	FILE* fp = fopen(filename, "w");

	if(fp == NULL) {
		return;
	}

    fprintf(fp, "digraph G {\n\n");

    for(int32_t i=0; i<g->vrt; i++) {

        fprintf(fp, "\t%d [label=\"%d\"];\n", i, i);

    }

    fprintf(fp, "\n");

    for(int32_t i=0; i<g->vrt; i++) {

        llist_t* edge = g->lst[i];

        while(edge != NULL) {

            int32_t dest = edge->dest;
            float w    = edge->weight;

            fprintf(fp, "\t%d -> %d [label=\"%f\"];\n", i, dest, w);  
            edge = edge->next;

        }

    }

    fprintf(fp, "\n}");

    fclose(fp);

}


void destroy_graph(graph_t* g) {

    if(g == NULL) {
        return;
    }

    for(int32_t i=0; i<g->vrt; i++) {

        destroy_llist(g->lst[i]);
        g->lst[i] = NULL;

    }

    g->max = 0;
    g->vrt = 0;
    
    free(g->deg);
    g->deg = NULL;

    free(g->lst);
    g->lst = NULL;

    free(g);

}

