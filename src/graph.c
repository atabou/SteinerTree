
#include <stdio.h>
#include <stdlib.h>

#include "pair.h"
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

    if(g->vrt < UINT32_MAX) {

        if(g->max == 0) {

            g->deg = (uint32_t*) malloc(sizeof(uint32_t));
            g->lst = (llist_t**) malloc(sizeof(llist_t*));

            g->max = 1;

        } else if(g->max * 2 < g->max) {

            g->deg = (uint32_t*) realloc(g->deg, sizeof(uint32_t) * UINT32_MAX);
            g->lst = (llist_t**) realloc(g->lst, sizeof(llist_t*) * UINT32_MAX);

            g->max = UINT32_MAX;

        } else if(g->vrt >= g->max) {

            g->deg = (uint32_t*) realloc(g->deg, sizeof(uint32_t) * 2 * g->max);
            g->lst = (llist_t**) realloc(g->lst, sizeof(llist_t*) * 2 * g->max);

            g->max = 2 * g->max;

        }

        g->deg[g->vrt]  = 0;
        g->lst[g->vrt]  = NULL;

        g->vrt = g->vrt + 1;

    }

    return g->vrt - 1;

}

void insert_edge(graph_t* g, uint32_t src, uint32_t dest, uint32_t w) {

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

    for(uint32_t i=0; i<g->vrt; i++) {

        fprintf(fp, "\t%d [label=\"%d\"];\n", i, i);

    }

    fprintf(fp, "\n");

    for(uint32_t i=0; i<g->vrt; i++) {

        llist_t* edge = g->lst[i];

        while(edge != NULL) {

            uint32_t dest = edge->dest;
            uint32_t w    = edge->weight;

            fprintf(fp, "\t%d -> %d [label=\"%d\"];\n", i, dest, w);  
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

    for(uint32_t i=0; i<g->vrt; i++) {

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



// O(max_id + V + E)
// graph* graph_union(graph* g1, graph* g2) {

//     graph* g = make_graph( max(g1->max_id, g2->max_id) );

//     for(int i=0; i<g1->nVertices; i++) {
//         insert_vertex(g, g1->hash[i]);
//     }

//     for(int i=0; i<g1->nVertices; i++) {

//         llist* e = g1->lst[i];
        
//         while(e != NULL) {

//             pair* p = (pair*) e->data;

//             int dest = g1->hash[(int) p->first];
//             int w    = (int) p->second;

//             insert_edge(g, g1->hash[i], dest, w);
//             e = e->next;

//         }

//     }

//     for(int j=0; j<g2->nVertices; j++) {
//         insert_vertex(g, g2->hash[j]);
//     }

//     for(int i=0; i<g2->nVertices; i++) {

//         llist* e = g2->lst[i];
        
//         while(e != NULL) {

//             pair* p = (pair*) e->data;

//             int dest = g2->hash[(int) p->first];
//             int w    = (int) p->second;

//             insert_edge(g, g2->hash[i], dest, w);
//             e = e->next;
            
//         }

//     }

//     return g;

// }




