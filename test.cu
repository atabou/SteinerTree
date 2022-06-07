
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "graph.hpp"
#include "query.hpp"
#include "table.hpp"
#include "util.hpp"
clock_t CLOCKMACRO;
#include "steiner.hpp"
#include "shortestpath.hpp"

#define NORMAL_COLOR  "\x1B[0m"
#define GREEN  "\x1B[32m"
#define BLUE  "\x1B[34m"
#define RED   "\033[31m"

void make_basic_graph(graph::graph_t** graph) {
    
    graph::make(graph);

    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);
    graph::insert_vertex(*graph);

    graph::insert_edge(*graph, 0, 1, 1);
    graph::insert_edge(*graph, 0, 1, 1);
    graph::insert_edge(*graph, 0, 2, 1);
    graph::insert_edge(*graph, 0, 3, 1);
    graph::insert_edge(*graph, 0, 4, 1);
    graph::insert_edge(*graph, 1, 5, 1);
    graph::insert_edge(*graph, 1, 6, 1);
    graph::insert_edge(*graph, 1, 8, 1);
    graph::insert_edge(*graph, 2, 5, 1);
    graph::insert_edge(*graph, 2, 7, 1);
    graph::insert_edge(*graph, 3, 7, 1);
    graph::insert_edge(*graph, 3, 9, 1);
    graph::insert_edge(*graph, 4, 6, 1);
    graph::insert_edge(*graph, 4, 9, 1);

    graph::insert_edge(*graph, 1, 0, 1);
    graph::insert_edge(*graph, 1, 0, 1);
    graph::insert_edge(*graph, 2, 0, 1);
    graph::insert_edge(*graph, 3, 0, 1);
    graph::insert_edge(*graph, 4, 0, 1);
    graph::insert_edge(*graph, 5, 1, 1);
    graph::insert_edge(*graph, 6, 1, 1);
    graph::insert_edge(*graph, 8, 1, 1);
    graph::insert_edge(*graph, 5, 2, 1);
    graph::insert_edge(*graph, 7, 2, 1);
    graph::insert_edge(*graph, 7, 3, 1);
    graph::insert_edge(*graph, 9, 3, 1);
    graph::insert_edge(*graph, 6, 4, 1);
    graph::insert_edge(*graph, 9, 4, 1);

}

void make_basic_query(query::query_t** terms) {

    query::make(terms);

    query::insert(*terms, 0);
    query::insert(*terms, 5);
    query::insert(*terms, 6);
    query::insert(*terms, 7);
    query::insert(*terms, 8);
    query::insert(*terms, 9);

}

void basictest() {

    graph::graph_t* graph = NULL;
    query::query_t* terms = NULL;
    
    make_basic_graph(&graph);
    make_basic_query(&terms);

    table::table_t< float >* dists = NULL;
    table::table_t<int32_t>* preds = NULL;

    table::make(&dists, graph->vrt, graph->vrt); 
    table::make(&preds, graph->vrt, graph->vrt);
       
    apsp_gpu_graph(graph, dists, preds);

    steiner::result_t* result1 = NULL;

    steiner_tree_cpu(graph, terms, dists, &result1);

    cudagraph::graph_t*        cuda_graph = NULL;
    cudatable::table_t<float>* cuda_dists = NULL;
    cudaquery::query_t*        cuda_terms = NULL;

    cudagraph::transfer_to_gpu(&cuda_graph, graph);
    cudatable::transfer_to_gpu(&cuda_dists, dists);
    cudaquery::transfer_to_gpu(&cuda_terms, terms);

    steiner::result_t* result2 = NULL;

    steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terms, terms->size, cuda_dists, preds, &result2);

    steiner::backtrack(terms, result2); 
    steiner::branch_and_clean(preds, result2);
    steiner::build_subgraph(graph, result2);

    cudatable::destroy(cuda_dists);
    cudagraph::destroy(cuda_graph);
    cudaquery::destroy(cuda_terms);

    graph::destroy(graph);
    query::destroy(terms);
    table::destroy(dists);
    table::destroy(preds);

}

void load_gr_file(char* filename, graph::graph_t** g, query::query_t** t, int32_t** h, int32_t* hsize, float* opt) {

    FILE* fp = fopen(filename, "r");

    printf("%s\n", filename);

    if(fp == NULL) {

        *g = NULL;
        *t = NULL;
        *h = NULL;
        return;

    }

    graph::make(g);
    query::make(t);

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
                query::insert(*t, (*h)[val]);

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

steiner::result_t* run(graph::graph_t* graph, query::query_t* terminals, table::table_t<float>** distances, table::table_t<int32_t>** parents, bool gpu) {

    if(*distances == NULL) { // All pairs shortest path.

        table::make(distances, graph->vrt, graph->vrt); 
        table::make(parents, graph->vrt, graph->vrt);

        apsp_gpu_graph(graph, *distances, *parents);

    }

    steiner::result_t* opt;

    if(gpu) {

        cudagraph::graph_t*        cuda_graph = NULL; 
        cudatable::table_t<float>* cuda_distances = NULL;
        cudaquery::query_t*        cuda_terminals = NULL;

        cudagraph::transfer_to_gpu(&cuda_graph, graph);
        cudatable::transfer_to_gpu(&cuda_distances, *distances);
        cudaquery::transfer_to_gpu(&cuda_terminals, terminals);

        steiner_tree_gpu(cuda_graph, graph->vrt, cuda_terminals, terminals->size, cuda_distances, *parents, &opt);

        steiner::backtrack(terminals, opt);
        steiner::branch_and_clean(*parents, opt);
        steiner::build_subgraph(graph, opt);

        cudaquery::destroy(cuda_terminals);
        cudatable::destroy(cuda_distances);
        cudagraph::destroy(cuda_graph);

    } else {

        steiner_tree_cpu(graph, terminals, *distances, &opt);

    }

    return opt;

}

void test(char* path) {

    DIR * d = opendir(path); // open the path

    struct dirent * dir; // for the directory entries

    while ((dir = readdir(d)) != NULL) {

        if(dir->d_type != DT_DIR) { // if the type is not directory && the file ends with .stp.       

            printf("%s%s\n",NORMAL_COLOR, dir->d_name);

            graph::graph_t* graph;
            query::query_t* terminals;
            int32_t* h;
            int32_t hsize;
            float expected;

            char filename[255]; // here I am using sprintf which is safer than strcat
            sprintf(filename, "%s/%s", path, dir->d_name);

            load_gr_file(filename, &graph, &terminals, &h, &hsize, &expected);

            table::table_t< float >* distances = NULL;
            table::table_t<int32_t>* predecessors = NULL;

            steiner::result_t* opt = run(graph, terminals, &distances, &predecessors, true);
            
            float value = opt->cost; 

            table::destroy(distances);
            table::destroy(predecessors);
            
            steiner::destroy(opt);

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

    /* basictest(); */
    test("./test");

}

