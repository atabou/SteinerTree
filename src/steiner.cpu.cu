
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <queue>
#include <unordered_map>

#include "steiner.hpp"
#include "combination.hpp"
#include "util.hpp"
#include "tree.hpp"


void fill_steiner_dp_table_cpu( graph::graph_t* g, query::query_t* terminals, table::table_t<float>* distances, table::table_t<float>* costs, table::table_t<int32_t>* roots, table::table_t<int64_t>* trees) {

    for(int32_t k=1; k <= terminals->size; k++) {

        uint64_t mask = 0;

        while( next_combination(terminals->size, k, &mask) ) { // while loop runs T choose k times (nCr).

            for(int32_t v=0; v < costs->n; v++) {

                if(k == 1) {

                    int32_t u = terminals->vals[terminals->size - __builtin_ffsll(mask)];

                    costs->vals[v * costs->m + (mask - 1)] = distances->vals[v * distances->m + u];
                    roots->vals[v * costs->m + (mask - 1)] = u;
                    trees->vals[v * costs->m + (mask - 1)] = -1; 

                } else {
                    
                    float   min_cost = FLT_MAX;
                    int32_t min_root = -1;
                    int64_t min_tree = -1;
                    
                    for(int32_t w=0; w < costs->n; w++) { // O(T * 2^T * (V+E))

                        if( query::element_exists(w, terminals, mask) ) { // O(V + E)

                            uint64_t submask = 1ll << (terminals->size - query::find_position(terminals, w) - 1);

                            float cost = distances->vals[v * distances->m + w] 
                                       + costs->vals[w * costs->m + ((mask & ~submask) - 1)];

                            if(cost < min_cost) {

                                min_cost = cost;
                                min_root = w;
                                min_tree = mask * ~submask;

                            }

                        } else if(g->deg[w] >= 3) { // O(2^T (V+E))

                            for(uint64_t submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask O(2^T)

                                float cost = distances->vals[v * distances->m + w] 
                                           + costs->vals[w * costs->m + submask - 1] 
                                           + costs->vals[w * costs->m + (mask & ~submask) - 1];

                                if(cost < min_cost) {

                                    min_cost = cost;
                                    min_root = w;
                                    min_tree = submask;

                                }

                            }

                        }

                    }

                    costs->vals[v * costs->m + mask - 1] = min_cost;
                    roots->vals[v * costs->m + mask - 1] = min_root;
                    trees->vals[v * costs->m + mask - 1] = min_tree;

                }
                
            }

        }

    }

}


void steiner::fill(graph::graph_t* g, query::query_t* terminals, table::table_t<float>* distances, steiner::result_t** result) {

    // Declare DP tables

    table::table_t< float >* costs = NULL;
    table::table_t<int32_t>* roots = NULL;
    table::table_t<int64_t>* trees = NULL;

    // Initialize DP tables.
    
    table::make(&costs, g->vrt, (int32_t) pow(2, terminals->size) - 1);
    table::make(&roots, g->vrt, (int32_t) pow(2, terminals->size) - 1);
    table::make(&trees, g->vrt, (int32_t) pow(2, terminals->size) - 1);

    // Fill dp table.

    fill_steiner_dp_table_cpu(g, terminals, distances, costs, roots, trees);

    // Initialize the steiner tree pointer.

    *result = (steiner::result_t*) malloc(sizeof(steiner::result_t));

    // Assign tables to result

    (*result)->costs = costs;
    (*result)->roots = roots;
    (*result)->trees = trees;

    // Calculate minimum from table.

    (*result)->cost = FLT_MAX;

    for(int32_t i=costs->m - 1; i < costs->m * distances->n; i+=costs->m) {
        
        if(costs->vals[i] < (*result)->cost) {

            (*result)->cost = costs->vals[i];
            (*result)->root = i;
            (*result)->tree = (*result)->costs->m;

        }

    } 

}


void backtrack_imp(query::query_t* terminals, steiner::result_t* result, int32_t start_root, int64_t start_tree, tree::tree_t* tree) {

    // Get the root and subtrees backlinks from the result structure

    int32_t root = result->roots->vals[result->roots->m * (start_tree - 1) + start_root];
    int32_t path = result->trees->vals[result->trees->m * (start_tree - 1) + start_root];

    // Initialize a subtree to represent the current vertex.

    tree::tree_t* subtree;
    tree::make(&subtree, root);

    // Set the initialized subtree as the child of the previously processed vertex.

    tree->subtrees.push_back(subtree);

    // Recursively add the subtrees of the vertex being processed currently. 

    if(path > 0 && query::element_exists(root, terminals, start_tree)) {

        backtrack_imp(terminals, result, root, path, subtree);
    
    } else if(path > 0) {

        backtrack_imp(terminals, result, root, path, subtree);
        backtrack_imp(terminals, result, root, start_tree & ~path, subtree);
    
    }   

}


void steiner::backtrack(query::query_t* terminals, steiner::result_t* result) {

    // Initialize the tree contained within the result structure

    tree::tree_t* mst = NULL;

    tree::make(&mst, result->root);

    result->mst = mst;
    
    // Construct the whole minimum steiner tree

    backtrack_imp(terminals, result, result->root, result->tree, result->mst);


}


void steiner::branch_and_clean(table::table_t<int32_t>* predecessors, steiner::result_t* result) {

    // Initialize a queue for bfs over the tree in result

    std::queue<tree::tree_t*> bfs = std::queue<tree::tree_t*>();

    // Push the root of the tree to the queue

    bfs.push(result->mst);

    // Run BFS

    while(!bfs.empty()) {

        tree::tree_t* current = bfs.front();
        
        bfs.pop();

        for(int i=0; i<current->subtrees.size(); i++) {

            if(current->subtrees[i]->vertex != current->vertex) {

                bfs.push(current->subtrees[i]);           
                
                int32_t pred = predecessors->vals[current->vertex * predecessors->m + current->subtrees[i]->vertex];
                tree::tree_t* child = current->subtrees[i];

                while(pred != current->vertex) {

                    tree::tree_t* path;

                    tree::make(&path, pred); 
                    path->subtrees.push_back(child);

                    pred = predecessors->vals[current->vertex * predecessors->m + pred];
                    child = path;

                }

                current->subtrees[i] = child;

            } else {

                tree::tree_t* tmp = current->subtrees[i];

                if(current->subtrees[i]->subtrees.size() == 0) {

                    current->subtrees.erase(current->subtrees.begin() + i);

                } else {

                    for(int j=1; j < current->subtrees[i]->subtrees.size(); j++) {

                        current->subtrees.push_back(current->subtrees[i]->subtrees[j]);

                    }

                    current->subtrees[i] = current->subtrees[i]->subtrees[0];
                
                }

                delete tmp;
                
                i--;
                 
            }

        }

    }

}


void steiner::build_subgraph(graph::graph_t* graph, steiner::result_t* result) {
   
    // Declare required data structures

    std::queue<tree::tree_t*> bfs;
    std::unordered_map<int32_t, int32_t> vrt;

    // Initialize structures

    graph::make(&(result->subgraph));
    
    vrt = std::unordered_map<int32_t, int32_t>();
    bfs = std::queue<tree::tree_t*>();
    
    // Pre-BFS preparations

    graph::insert_vertex(result->subgraph);
    
    vrt.insert({result->mst->vertex, 0});
    bfs.push(result->mst);

    while(!bfs.empty()) {

        tree::tree_t* current = bfs.front();
        
        bfs.pop();

        for(int i=0; i<current->subtrees.size(); i++) {  
            
            int32_t position = graph::insert_vertex(result->subgraph);
            vrt.insert({current->subtrees[i]->vertex, position});
            bfs.push(current->subtrees[i]);

            float wgt = graph::weight(graph, current->vertex, current->subtrees[i]->vertex);

            graph::insert_edge(result->subgraph, vrt[current->vertex], vrt[current->subtrees[i]->vertex], wgt);
            graph::insert_edge(result->subgraph, vrt[current->subtrees[i]->vertex], vrt[current->vertex], wgt);
                        
        }

    }

}


void steiner::destroy(steiner::result_t* result) {

    table::destroy(result->costs);
    table::destroy(result->roots);
    table::destroy(result->trees);
    tree::destroy(result->mst);
    graph::destroy(result->subgraph);

    free(result);

}


