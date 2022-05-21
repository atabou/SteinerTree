
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.h"

graph_t* make_graph() {

    graph_t* g = (graph_t*) malloc(sizeof(graph_t));

    g->max = 0;
    g->vrt = 0;
    
    g->deg = NULL;
    
    g->dst = NULL;    
    g->wgt = NULL;

    return g;

}

uint32_t insert_vertex(graph_t* g) {

    if(g->vrt < INT32_MAX) {

        if(g->max == 0) {

            g->deg =  (int32_t*) malloc(sizeof(int32_t));
            
            g->dst = (int32_t**) malloc(sizeof(int32_t*));
            g->wgt = ( float** ) malloc(sizeof( float* ));

            g->max = 1;

        } else if(g->max * 2 < g->max) { // Executes if there is an overflow

            g->deg =  (int32_t*) realloc(g->deg, sizeof(int32_t) * INT32_MAX);
            
            g->dst = (int32_t**) realloc(g->dst, sizeof(int32_t*) * INT32_MAX);
            g->wgt = ( float** ) realloc(g->wgt, sizeof( float* ) * INT32_MAX);

            g->max = INT32_MAX;

        } else if(g->vrt >= g->max) {

            g->deg = (int32_t*) realloc(g->deg, sizeof(int32_t) * 2 * g->max);

            g->dst = (int32_t**) realloc(g->dst, sizeof(int32_t*) * 2 * g->max);
            g->wgt = ( float** ) realloc(g->wgt, sizeof( float* ) * 2 * g->max);


            g->max = 2 * g->max;

        }

        g->deg[g->vrt] = 0;
        g->dst[g->vrt] = NULL;
        g->wgt[g->vrt] = NULL;

        g->vrt = g->vrt + 1;

    }

    return g->vrt - 1;

}

void insert_edge(graph_t* g, int32_t src, int32_t dest, float w) {

    if(src < g->vrt && dest < g->vrt && g->deg[src] < INT32_MAX) {

        for(int32_t i=0; i<g->deg[src]; i++) {
            if(dest == g->dst[src][i]) {
                return;
            }
        }

        if(g->deg[src] == 0) {

            g->dst[src] = (int32_t*) malloc(sizeof(int32_t));
            g->wgt[src] = ( float* ) malloc(sizeof( float )); 

            g->dst[src][0] = dest;
            g->wgt[src][0] = w;

            g->deg[src] = 1;

        } else {

            g->dst[src] = (int32_t*) realloc(g->dst[src], sizeof(int32_t) * (g->deg[src] + 1));
            g->wgt[src] = ( float* ) realloc(g->wgt[src], sizeof( float ) * (g->deg[src] + 1));

            g->dst[src][g->deg[src]] = dest;
            g->wgt[src][g->deg[src]] = w;

            g->deg[src] = g->deg[src] + 1;

        }
        
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

        for(int32_t j=0; j<g->deg[i]; j++) {

            int32_t dest = g->dst[i][j];
            float w    = g->wgt[i][j];

            fprintf(fp, "\t%d -> %d [label=\"%f\"];\n", i, dest, w);  

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
        
        free(g->dst[i]);
        free(g->wgt[i]);

    }

    g->max = 0;
    g->vrt = 0;
    
    free(g->deg);
    g->deg = NULL;

    free(g->dst);
    g->dst = NULL;

    free(g->wgt);
    g->wgt = NULL;

    free(g);

}

