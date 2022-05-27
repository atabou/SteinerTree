
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

#include "util.h"
clock_t CLOCKMACRO;

#include "graph.h"
#include "shortestpath.h"
#include "steiner.h"
#include "set.h"
#include "table.h"

#include "graph.cuda.h"


graph_t* make_randomly_connected_graph(uint32_t v) {

    graph_t* g = make_graph();

    for(int i=0; i<v; i++) {
        insert_vertex(g);
    }

    int used[v];

    for(int i=0; i<v; i++) {

        for(int j=0; j<v; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<v; j++) { // Shuffle array

            int x = rand() % v;
            int y = rand() % v;

            int tmp = used[x];
            used[x] = used[y];
            used[y] = tmp;

        }

        int deg = rand() % v;

        for(int j=0; j<deg; j++) {

            if(used[j] != -1) {

                insert_edge(g, i, used[j], 1);
                insert_edge(g, used[j], i, 1);

            }

        }

    }

    return g;

}

/*
void specify_args(int argc, char** argv) {

    uint32_t V = atoi(argv[1]);
    uint32_t T = atoi(argv[2]);

    graph_t* g = make_randomly_connected_graph(V);

    set_t* t = make_set();

    for(int i=0; i<T; i++) {

        set_insert(t, rand() % (V + 1) );

    }
    
    table_t* steiner = steiner_tree(g, t);
    
    destroy_set(t);
    destroy_graph(g);
    free_table(steiner);

}
*/

void run(graph_t* graph, set_t* terminals, table_t** distances, table_t** parents, bool cpu, bool gpu) {
 
    printf("|V|= %u, |T|= %u:\n", graph->vrt, terminals->size);

    if(*distances == NULL) { // All pairs shortest path.
      
        *distances = make_table(graph->vrt, graph->vrt); 
        *parents   = make_table(graph->vrt, graph->vrt);
       
        TIME(apsp_gpu_graph(graph, *distances, *parents), "\tAPSP:");

    }

    if(cpu) {
        steiner_tree_cpu(graph, terminals, *distances);
    }

    cudagraph_t* cuda_graph     = copy_cudagraph(graph);
    cudaset_t*   cuda_terminals = copy_cudaset(terminals);
    cudatable_t* cuda_distances = copy_cudatable(*distances);
    
    if(gpu) {
        steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances);
    }

    printf("\n");

    free_cudatable(cuda_distances);
    free_cudaset(cuda_terminals);
    free_cudagraph(cuda_graph);
    
}

int main(int argc, char** argv) {
    
    cudagraph_t* graph;
    cudaset_t* terminals;
    cudatable_t* distances = NULL;
    cudatable_t* predecessors = NULL;

    for(int32_t vrt=512; vrt <= 65536; vrt = vrt * 2) {

        graph = make_randomly_connected_graph(vrt);

        printf("%d\n", graph->vrt);

        for(int32_t t=2; t < 10; t++) {
            
            terminals = make_set();

            for(int i=0; i<t; i++) {
                set_insert(terminals, rand() % (vrt + 1) );
            }

            run(graph, terminals, &distances, &predecessors, false, true);

            free_set(terminals);

        }

        destroy_graph(graph);
        free_table(distances);
        distances = NULL;
        free_table(predecessors);
        predecessors = NULL;

    }
    
    return 0;
	
}
