
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph.hpp"

void graph::make(graph::graph_t** g) {

    *g = (graph::graph_t*) malloc(sizeof(graph::graph_t));

    (*g)->max = 0;
    (*g)->vrt = 0;
    
    (*g)->deg = NULL;
    
    (*g)->dst = NULL;    
    (*g)->wgt = NULL;

}

int32_t graph::insert_vertex(graph::graph_t* g) {

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

void graph::insert_edge(graph::graph_t* g, int32_t src, int32_t dest, float w) {

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

__host__ float graph::weight(graph_t* graph, int32_t src, int32_t dst) {

    if(src >= 0 && dst >= 0 && src < graph->vrt && src < graph->vrt) {

        for(int i=0; i<graph->deg[src]; i++) {

            if(graph->dst[src][i] == dst) {

                return graph->wgt[src][i];

            }

        }

    }

    return NAN;

}

void graph::to_graphviz(graph::graph_t* g, char* filename) {

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


void graph::destroy(graph::graph_t* g) {

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


graph::graph_t* make_cudagraph() {

    cudaError_t err;

    graph::graph_t* graph = NULL;

    err = cudaMalloc(&graph, sizeof(graph::graph_t));

    if(err) {
        printf("Error initializing memory for cuda graph. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after graph memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    return graph;

}

void cudagraph::transfer_to_gpu(cudagraph::graph_t** graph_d, graph::graph_t* graph) {

    // Initialize cuda graph

    *graph_d = (cudagraph::graph_t*) malloc(sizeof(cudagraph::graph_t));
    (*graph_d)->vrt = graph->vrt;

    cudaError_t err;
    graph::graph_t tmp;

    // Set cuda graph max capacity

    tmp.max = graph->vrt;

    // Set cuda graph number of vertices

    tmp.vrt = graph->vrt;

    // Create and set cuda graph degree array

    err = cudaMalloc(&(tmp.deg), sizeof(int32_t) * graph->vrt);
    
    if(err) { 
        printf("Could not allocate memory for graph degree array. (Error code: %d)\n", err);
        exit(err);
    }

    // Create and set cuda graph adjacency array

    err = cudaMalloc(&(tmp.dst), sizeof(int32_t*) * graph->vrt);
    err = cudaMalloc(&(tmp.wgt), sizeof( float* ) * graph->vrt);

    if(err) {
        printf("Could not allocate memory for graph adjacency lists. (Error code: %d)\n", err);
        exit(err);
    }

    // Synchronize after memory allocation.

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    // Copy degrees into degree array

    cudaMemcpy(tmp.deg, graph->deg, sizeof(int32_t) * graph->vrt, cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after degree memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Create and set cuda graph adjacency list

    int32_t** dst[graph->vrt];
    float**   wgt[graph->vrt];

    for(int32_t i=0; i<graph->vrt; i++) {

        cudaMemcpy(&dst[i], graph->dst[i], sizeof(int32_t) * graph->deg[i], cudaMemcpyHostToDevice);
        cudaMemcpy(&wgt[i], graph->wgt[i], sizeof( float ) * graph->deg[i], cudaMemcpyHostToDevice);

    }

    cudaMemcpy(tmp.dst, dst, sizeof(int32_t*) * graph->vrt, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp.wgt, wgt, sizeof( float* ) * graph->vrt, cudaMemcpyHostToDevice);
    
    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after dst and wgt copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Copy cuda graph struct to gpu

    (*graph_d)->graph = make_cudagraph();

    cudaMemcpy((*graph_d)->graph, &tmp, sizeof(graph::graph_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after graph memory copy. (Error code: %d)\n", err);
        exit(err);
    }
    
    cudaMemcpy(&tmp, (*graph_d)->graph, sizeof(graph::graph_t), cudaMemcpyDeviceToHost);
    
}

void cudagraph::destroy(cudagraph::graph_t* g) {
 
    cudaError_t err;
    graph::graph_t tmp;

    cudaMemcpy(&tmp, g->graph, sizeof(graph::graph_t), cudaMemcpyDeviceToHost);
 
    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy graph data before free. (Error code: %d)\n", err);
        exit(err);
    }
    
    err = cudaFree(g->graph);

    if(err) {
        printf("Could not deallocate cuda graph structure. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaFree(tmp.deg);

    if(err) {
        printf("Could not deallocate cuda degree array. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda degree array free. (Error code: %d)\n", err);
        exit(err);
    }

    int32_t** del1[tmp.vrt];
    float**   del2[tmp.vrt];

    cudaMemcpy(del1, tmp.dst, sizeof(int32_t*) * tmp.vrt, cudaMemcpyDeviceToHost);
    cudaMemcpy(del2, tmp.wgt, sizeof( float* ) * tmp.vrt, cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda graph memcpy from device. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaFree(tmp.dst);
    err = cudaFree(tmp.wgt);
    
    if(err) {
        printf("Could not deallocate cuda lst arrays. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda graph lst free from device. (Error code: %d)\n", err);
        exit(err);
    }
 
    for(int32_t i=0; i<tmp.vrt; i++) {

        cudaFree(del1[i]);
        cudaFree(del2[i]);
    
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after sub-list free from device. (Error code: %d)\n", err);
        exit(err);
    }

    free(g);
    
}



