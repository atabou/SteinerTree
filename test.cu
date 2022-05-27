
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

graph_t* test_graph1() {

    graph_t* g = make_graph();

    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);

    insert_edge(g, 0, 1, 1);
    insert_edge(g, 0, 1, 1);
    insert_edge(g, 0, 2, 1);
    insert_edge(g, 0, 3, 1);
    insert_edge(g, 0, 4, 1);
    insert_edge(g, 1, 5, 1);
    insert_edge(g, 1, 6, 1);
    insert_edge(g, 1, 8, 1);
    insert_edge(g, 2, 5, 1);
    insert_edge(g, 2, 7, 1);
    insert_edge(g, 3, 7, 1);
    insert_edge(g, 3, 9, 1);
    insert_edge(g, 4, 6, 1);
    insert_edge(g, 4, 9, 1);

    insert_edge(g, 1, 0, 1);
    insert_edge(g, 1, 0, 1);
    insert_edge(g, 2, 0, 1);
    insert_edge(g, 3, 0, 1);
    insert_edge(g, 4, 0, 1);
    insert_edge(g, 5, 1, 1);
    insert_edge(g, 6, 1, 1);
    insert_edge(g, 8, 1, 1);
    insert_edge(g, 5, 2, 1);
    insert_edge(g, 7, 2, 1);
    insert_edge(g, 7, 3, 1);
    insert_edge(g, 9, 3, 1);
    insert_edge(g, 6, 4, 1);
    insert_edge(g, 9, 4, 1);
    return g;

}

void basictest() {

    graph_t* graph = test_graph1();
    set_t* terminals = make_set();

    set_insert(terminals, 0);
    set_insert(terminals, 5);
    set_insert(terminals, 6);
    set_insert(terminals, 7);
    set_insert(terminals, 8);
    set_insert(terminals, 9);

    table_t* distances = make_table(graph->vrt, graph->vrt); 
    table_t* parents   = make_table(graph->vrt, graph->vrt);
       
    apsp_gpu_graph(graph, distances, parents);

    steiner_tree_cpu(graph, terminals, distances);

    cudagraph_t* cuda_graph     = copy_cudagraph(graph);
    cudaset_t*   cuda_terminals = copy_cudaset(terminals);
    cudatable_t* cuda_distances = copy_cudatable(distances);

    steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances);

    free_cudatable(cuda_distances);
    free_cudaset(cuda_terminals);
    free_cudagraph(cuda_graph);

    destroy_graph(graph);
    free_set(terminals);
    free_table(distances);
    free_table(parents);

}

void load_gr_file(char* filename, graph_t** g, set_t** t, int32_t** h, int32_t* hsize, float* opt) {

    FILE* fp = fopen(filename, "r");

    printf("%s\n", filename);

    if(fp == NULL) {

        *g = NULL;
        *t = NULL;
        *h = NULL;
        return;

    }

    *t = make_set();
    *g = make_graph();

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
                set_insert(*t, (*h)[val]);

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

float run(graph_t* graph, set_t* terminals, table_t** distances, table_t** parents, bool gpu) {

    if(*distances == NULL) { // All pairs shortest path.

        *distances = make_table(graph->vrt, graph->vrt); 
        *parents   = make_table(graph->vrt, graph->vrt);

        apsp_gpu_graph(graph, *distances, *parents);

    }

    steiner_result opt;

    if(gpu) {

        cudagraph_t* cuda_graph     = copy_cudagraph(graph);
        cudaset_t*   cuda_terminals = copy_cudaset(terminals);
        cudatable_t* cuda_distances = copy_cudatable(*distances);

        opt = steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances);

        free_cudatable(cuda_distances);
        free_cudaset(cuda_terminals);
        free_cudagraph(cuda_graph);

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

            graph_t* graph;
            set_t* terminals;
            int32_t* h;
            int32_t hsize;
            float expected;

            char filename[255]; // here I am using sprintf which is safer than strcat
            sprintf(filename, "%s/%s", path, dir->d_name);

            load_gr_file(filename, &graph, &terminals, &h, &hsize, &expected);

            table_t* distances = NULL;
            table_t* predecessors = NULL;

            float value = run(graph, terminals, &distances, &predecessors, true);

            free_table(distances);
            free_table(predecessors);

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

