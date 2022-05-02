
#include <stdio.h>

extern "C" {
    #include "graph.cuda.h"
    #include "llist.cuda.h"
}

cudagraph_t* make_cudagraph() {

    cudaError_t err;

    cudagraph_t* cuda_graph = NULL;

    err = cudaMalloc(&cuda_graph, sizeof(cudagraph_t));

    if(err) {
        printf("Error initializing memory for cuda graph. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after graph memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    return cuda_graph;

}

cudagraph_t* copy_cudagraph(graph_t* cpu_graph) {

    cudaError_t err;
    cudagraph_t tmp;

    // Set cuda graph max capacity

    tmp.max = cpu_graph->max;

    // Set cuda graph number of vertices

    tmp.vrt = cpu_graph->vrt;

    // Create and set cuda graph degree array

    err = cudaMalloc(&(tmp.deg), sizeof(uint32_t) * cpu_graph->max);
    
    if(err) { 
        printf("Could not allocate memory for graph degree array. (Error code: %d)\n", err);
        exit(err);
    }

    // Create and set cuda graph adjacency array

    err = cudaMalloc(&(tmp.lst), sizeof(cudallist_t*) * cpu_graph->max);

    if(err) {
        printf("Could not allocate memory for graph adjacency list. (Error code: %d)\n", err);
        exit(err);
    }

    // Synchronize after memory allocation.

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after memory allocation. (Error code: %d)\n", err);
        exit(err);
    }

    // Copy degrees into degree array

    cudaMemcpy(tmp.deg, cpu_graph->deg, sizeof(uint32_t) * cpu_graph->vrt, cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after degree memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Create and set cuda graph adjacency list

    cudallist_t* edges[cpu_graph->vrt];

    for(uint32_t i=0; i<cpu_graph->vrt; i++) {

        edges[i] = copy_cudallist(cpu_graph->lst[i], cpu_graph->deg[i]);

    }

    cudaMemcpy(tmp.lst, edges, sizeof(cudallist_t*) * cpu_graph->vrt, cudaMemcpyHostToDevice);
 
    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after llist array memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    // Copy cuda graph struct to gpu

    cudagraph_t* cuda_graph = make_cudagraph();

    cudaMemcpy(cuda_graph, &tmp, sizeof(cudagraph_t), cudaMemcpyHostToDevice);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize cuda device after graph memory copy. (Error code: %d)\n", err);
        exit(err);
    }

    return cuda_graph;

}

void free_cudagraph(cudagraph_t* g) {
 
    cudaError_t err;
    cudagraph_t tmp;

    cudaMemcpy(&tmp, g, sizeof(cudagraph_t), cudaMemcpyDeviceToHost);
 
    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not copy graph data before free. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaFree(g);

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

    cudallist_t* del[tmp.vrt];

    cudaMemcpy(del, tmp.lst, sizeof(cudallist_t*) * tmp.vrt, cudaMemcpyDeviceToHost);

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda graph memcpy from device. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaFree(tmp.lst);

    if(err) {
        printf("Could not deallocate cuda lst array. (Error code: %d)\n", err);
        exit(err);
    }

    err = cudaDeviceSynchronize();

    if(err) {
        printf("Could not synchronize after cuda graph lst free from device. (Error code: %d)\n", err);
        exit(err);
    }
 
    for(uint32_t i=0; i<tmp.vrt; i++) {
        free_cudallist(del[i]);
    }
    
}


