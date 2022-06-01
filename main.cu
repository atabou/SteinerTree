
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
#include "query.h"
#include "table.h"

graph::graph_t* make_randomly_connected_graph(uint32_t v) {

    graph::graph_t* g = NULL;
    
    graph::make(&g);

    for(int i=0; i<v; i++) {
        graph::insert_vertex(g);
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

                graph::insert_edge(g, i, used[j], 1);
                graph::insert_edge(g, used[j], i, 1);

            }

        }

    }

    return g;

}

/*
void specify_args(int argc, char** argv) {

    uint32_t V = atoi(argv[1]);
    uint32_t T = atoi(argv[2]);

    graph::graph_t* g = make_randomly_connected_graph(V);

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

void run(graph::graph_t* graph, query::query_t* terminals, table::table_t** distances, table::table_t** parents, bool cpu, bool gpu) {
 
    steiner_result* result = NULL;

    printf("|V|= %u, |T|= %u:\n", graph->vrt, terminals->size);

    if(*distances == NULL) { // All pairs shortest path.
      
        table::make(distances, graph->vrt, graph->vrt); 
        table::make(parents, graph->vrt, graph->vrt);
       
        TIME(apsp_gpu_graph(graph, *distances, *parents), "\tAPSP:");

    }

    if(cpu) {

        steiner_tree_cpu(graph, terminals, *distances, &result);

    }

    if(gpu) {

        cudagraph::graph_t* graph_d = NULL;
        cudaquery::query_t* terms_d = NULL;
        cudatable::table_t* dists_d = NULL;

        cudagraph::transfer_to_gpu(&graph_d, graph);
        cudaquery::transfer_to_gpu(&terms_d, terminals);
        cudatable::transfer_to_gpu(&dists_d, *distances);
    
        steiner_tree_gpu(graph_d, graph->vrt, terms_d, terminals->size, dists_d, &result);
    
        printf("\n");

        cudatable::destroy(dists_d);
        cudaquery::destroy(terms_d);
        cudagraph::destroy(graph_d);

    }
    
}

int main(int argc, char** argv) {
    
    graph::graph_t* graph = NULL;
    query::query_t* terms = NULL;
    table::table_t* dists = NULL;
    table::table_t* preds = NULL;

    for(int32_t vrt=8192; vrt <= 65536; vrt = vrt * 2) {

        graph = make_randomly_connected_graph(vrt);

        for(int32_t t=2; t < 10; t++) {
            
            query::make(&terms);

            for(int i=0; i<t; i++) {
                query::insert(terms, rand() % (vrt + 1) );
            }

            run(graph, terms, &dists, &preds, false, true);

            query::destroy(terms);

        }

        graph::destroy(graph);
        table::destroy(dists);
        table::destroy(preds);
        
        graph = NULL;
        dists = NULL;
        preds = NULL;

    }
    
    return 0;
	
}
