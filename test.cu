
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "graph.h"
#include "set.h"
#include "table.h"
#include "util.h"
clock_t CLOCKMACRO;
#include "steiner.h"
#include "shortestpath.h"

#define NORMAL_COLOR  "\x1B[0m"
#define GREEN  "\x1B[32m"
#define BLUE  "\x1B[34m"
#define RED   "\033[31m"

graph::graph_t* test_graph1() {

    graph::graph_t* g;
    
    graph::make(&g);

    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);
    graph::insert_vertex(g);

    graph::insert_edge(g, 0, 1, 1);
    graph::insert_edge(g, 0, 1, 1);
    graph::insert_edge(g, 0, 2, 1);
    graph::insert_edge(g, 0, 3, 1);
    graph::insert_edge(g, 0, 4, 1);
    graph::insert_edge(g, 1, 5, 1);
    graph::insert_edge(g, 1, 6, 1);
    graph::insert_edge(g, 1, 8, 1);
    graph::insert_edge(g, 2, 5, 1);
    graph::insert_edge(g, 2, 7, 1);
    graph::insert_edge(g, 3, 7, 1);
    graph::insert_edge(g, 3, 9, 1);
    graph::insert_edge(g, 4, 6, 1);
    graph::insert_edge(g, 4, 9, 1);

    graph::insert_edge(g, 1, 0, 1);
    graph::insert_edge(g, 1, 0, 1);
    graph::insert_edge(g, 2, 0, 1);
    graph::insert_edge(g, 3, 0, 1);
    graph::insert_edge(g, 4, 0, 1);
    graph::insert_edge(g, 5, 1, 1);
    graph::insert_edge(g, 6, 1, 1);
    graph::insert_edge(g, 8, 1, 1);
    graph::insert_edge(g, 5, 2, 1);
    graph::insert_edge(g, 7, 2, 1);
    graph::insert_edge(g, 7, 3, 1);
    graph::insert_edge(g, 9, 3, 1);
    graph::insert_edge(g, 6, 4, 1);
    graph::insert_edge(g, 9, 4, 1);

    return g;

}

void basictest() {

    graph::graph_t* graph = test_graph1();
    
    set::set_t* terminals = NULL;
    
    set::make(&terminals);

    set::insert(terminals, 0);
    set::insert(terminals, 5);
    set::insert(terminals, 6);
    set::insert(terminals, 7);
    set::insert(terminals, 8);
    set::insert(terminals, 9);

    table::table_t* distances = table::make(graph->vrt, graph->vrt); 
    table::table_t* parents   = table::make(graph->vrt, graph->vrt);
       
    apsp_gpu_graph(graph, distances, parents);

    steiner_tree_cpu(graph, terminals, distances);

    cudagraph::graph_t* cuda_graph = NULL;
    cudaset::set_t* cuda_terminals = NULL;

    cudagraph::transfer_to_gpu(&cuda_graph, graph);
    cudatable::table_t* cuda_distances = cudatable::transfer_to_gpu(distances);
    cudaset::transfer_to_gpu(&cuda_terminals, terminals);

    steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances);

    cudatable::destroy(cuda_distances);
    cudagraph::destroy(cuda_graph);
    cudaset::destroy(cuda_terminals);

    graph::destroy(graph);
    set::destroy(terminals);
    table::destroy(distances);
    table::destroy(parents);

}

void load_gr_file(char* filename, graph::graph_t** g, set::set_t** t, int32_t** h, int32_t* hsize, float* opt) {

    FILE* fp = fopen(filename, "r");

    printf("%s\n", filename);

    if(fp == NULL) {

        *g = NULL;
        *t = NULL;
        *h = NULL;
        return;

    }

    graph::make(g);
    set::make(t);

    *h = (int32_t*) malloc(sizeof(int32_t));
    (*h)[0] = INT32_MAX;
    *hsize = 1;

    char* line = NULL;
    size_t buff = 0;
    ssize_t len = 0;

    while( (len  = getline(&line, &buff, fp)) != -1 ) {

        char* token = strtok(line, " ");
        int type = 0;

        while(token != NULL) {

            if(type == 1) {

                int32_t x = atoi(token);

                if(x >= *hsize) {

                    *h = (int32_t*) realloc(*h, sizeof(int32_t) * (x + 1));

                    for(uint32_t i=*hsize; i < (x+1); i++) {
                        (*h)[i] = INT32_MAX;
                    }

                    *hsize = x + 1;

                }

                if( (*h)[x] == INT32_MAX ) {

                    int32_t id = insert_vertex(*g);
                    (*h)[x] = id; 

                }

                token = strtok(NULL, " ");
                int32_t y = atoi(token);

                if(y >= *hsize) {

                    *h = (int32_t*) realloc(*h, sizeof(int32_t) * (y + 1));

                    for(int32_t i=*hsize; i < (y+1); i++) {
                        (*h)[i] = INT32_MAX;
                    }

                    *hsize = y + 1;

                }

                if( (*h)[y] == INT32_MAX ) {

                    int32_t id = insert_vertex(*g);
                    (*h)[y] = id; 

                }

                token = strtok(NULL, " ");
                float w = atof(token);

                insert_edge(*g, (*h)[x], (*h)[y], w);
                insert_edge(*g, (*h)[y], (*h)[x], w);

            } else if(type == 2) {

                int32_t val = atoi(token);
                set::insert(*t, (*h)[val]);

            } else if(type == 3) {

                *opt = atof(token);

            }

            if(token[0] == 'E' && token[1] == '\0') {
                type = 1;
            } else if(token[0] == 'T' && token[1] == '\0') {
                type = 2;
            } else if(token[0] == 'O') {
                type = 3;
            }

            token = strtok(NULL, " ");

        }

    }

    fclose(fp);

}

float run(graph::graph_t* graph, set::set_t* terminals, table::table_t** distances, table::table_t** parents, bool gpu) {

    if(*distances == NULL) { // All pairs shortest path.

        *distances = table::make(graph->vrt, graph->vrt); 
        *parents   = table::make(graph->vrt, graph->vrt);

        apsp_gpu_graph(graph, *distances, *parents);

    }

    steiner_result opt;

    if(gpu) {

        cudagraph::graph_t* cuda_graph = NULL; 
        cudaset::set_t* cuda_terminals = NULL;

        cudagraph::transfer_to_gpu(&cuda_graph, graph);
        cudaset::transfer_to_gpu(&cuda_terminals, terminals);
        cudatable::table_t* cuda_distances = cudatable::transfer_to_gpu(*distances);

        opt = steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances);

        cudatable::destroy(cuda_distances);
        cudaset::destroy(cuda_terminals);
        cudagraph::destroy(cuda_graph);

    } else {

        opt = steiner_tree_cpu(graph, terminals, *distances);

    }

    return opt.cost;

}

void test(char* path) {

    DIR * d = opendir(path); // open the path

    struct dirent * dir; // for the directory entries

    while ((dir = readdir(d)) != NULL) {

        if(dir->d_type != DT_DIR) { // if the type is not directory && the file ends with .stp.       

            printf("%s%s\n",NORMAL_COLOR, dir->d_name);

            graph::graph_t* graph;
            set::set_t* terminals;
            int32_t* h;
            int32_t hsize;
            float expected;

            char filename[255]; // here I am using sprintf which is safer than strcat
            sprintf(filename, "%s/%s", path, dir->d_name);

            load_gr_file(filename, &graph, &terminals, &h, &hsize, &expected);

            table::table_t* distances = NULL;
            table::table_t* predecessors = NULL;

            float value = run(graph, terminals, &distances, &predecessors, true);

            table::destroy(distances);
            table::destroy(predecessors);

            if(fabs(value - expected) < 1e-4) {
                printf("%sTEST SUCCEEDED.%s\n", GREEN, NORMAL_COLOR);
            } else {
                printf("%sTEST FAILED - Expected: %f, Got: %f%s\n", RED, expected, value, NORMAL_COLOR);
            }

            printf("\n");

        } else if(dir -> d_type == DT_DIR && strcmp(dir->d_name,".")!=0 && strcmp(dir->d_name,"..")!=0 ) { // if it is a directory

            char d_path[255]; // here I am using sprintf which is safer than strcat
            sprintf(d_path, "%s/%s", path, dir->d_name);
            test(d_path); // recall with the new path

        }
    }

    closedir(d); // finally close the directory


}

int main() {

    // basictest();
    test("./test");

}

