
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "graph.h"
#include "pair.h"
#include "steiner.h"
#include "set.h"
#include "common.h"

graph* test_graph1() {

    int max_id = 10;

    graph* g = make_graph(max_id);

    insert_vertex(g, 1);
    insert_vertex(g, 2);
    insert_vertex(g, 3);
    insert_vertex(g, 4);
    insert_vertex(g, 5);
    insert_vertex(g, 6);
    insert_vertex(g, 7);
    insert_vertex(g, 8);
    insert_vertex(g, 9);
    insert_vertex(g, 10);

    insert_edge(g, 1, 2, 1);
    insert_edge(g, 1, 2, 1);
    insert_edge(g, 1, 3, 1);
    insert_edge(g, 1, 4, 1);
    insert_edge(g, 1, 5, 1);
    insert_edge(g, 2, 6, 1);
    insert_edge(g, 2, 7, 1);
    insert_edge(g, 2, 9, 1);
    insert_edge(g, 3, 6, 1);
    insert_edge(g, 3, 8, 1);
    insert_edge(g, 4, 8, 1);
    insert_edge(g, 4, 10, 1);
    insert_edge(g, 5, 7, 1);
    insert_edge(g, 5, 10, 1);

    insert_edge(g, 2, 1, 1);
    insert_edge(g, 2, 1, 1);
    insert_edge(g, 3, 1, 1);
    insert_edge(g, 4, 1, 1);
    insert_edge(g, 5, 1, 1);
    insert_edge(g, 6, 2, 1);
    insert_edge(g, 7, 2, 1);
    insert_edge(g, 9, 2, 1);
    insert_edge(g, 6, 3, 1);
    insert_edge(g, 8, 3, 1);
    insert_edge(g, 8, 4, 1);
    insert_edge(g,10, 4, 1);
    insert_edge(g, 7, 5, 1);
    insert_edge(g,10, 5, 1);

    return g;

}

    

graph* test_graph2() {

    graph* g = make_graph(7);

    insert_vertex(g, 0);
    insert_vertex(g, 1);
    insert_vertex(g, 2);
    insert_vertex(g, 3);
    insert_vertex(g, 4);
    insert_vertex(g, 5);
    insert_vertex(g, 6);

    insert_edge(g, 0, 1, 1);
    insert_edge(g, 1, 0, 1);

    insert_edge(g, 1, 2, 1);
    insert_edge(g, 2, 1, 1);

    insert_edge(g, 2, 3, 1);
    insert_edge(g, 3, 2, 1);

    insert_edge(g, 1, 4, 1);
    insert_edge(g, 4, 1, 1);

    insert_edge(g, 2, 4, 1);
    insert_edge(g, 4, 2, 1);

    insert_edge(g, 4, 5, 1);
    insert_edge(g, 5, 4, 1);

    insert_edge(g, 4, 6, 1);
    insert_edge(g, 6, 4, 1);

    return g;

}

/**
 * @brief Makes a randomly connected graph with a maximum.
 * 
 * @param max_id the highest id that will be used to represent vertices in this graph.
 * @return graph* a randomly connected graph.
 */
graph* make_randomly_connected_graph(int max_id) {

    graph* g = make_graph(max_id);

    for(int i=0; i<g->max_id + 1; i++) {
        insert_vertex(g, i);
    }

    int used[max_id + 1];

    for(int i=0; i<max_id+1; i++) {

        for(int j=0; j<max_id + 1; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<max_id + 1; j++) { // Shuffle array

            int x = rand() % (max_id + 1);
            int y = rand() % (max_id + 1);

            int tmp = used[x];
            used[x] = used[y];
            used[y] = tmp;

        }

        int deg = rand() % max_id;

        for(int j=0; j<deg; j++) {

            if(used[j] != -1) {

                insert_edge(g, i, used[j], 1);
                insert_edge(g, used[j], i, 1);

            }

        }

    }

    return g;

}

int main(int argc, char** argv) {
    
    // graph* g = test_graph1();

    // to_graphviz(g, "test.dot");
    
    // set_t* t = make_set();

    // // set_insert(t, 3);
    // // set_insert(t, 5);

    // set_insert(t, 1);
    // set_insert(t, 6);
    // set_insert(t, 7);
    // set_insert(t, 8);
    // set_insert(t, 9);
    // set_insert(t, 10);
    
    // pair* steiner = steiner_tree(g, t);
    
    // graph* tree = (graph*) steiner->first;
    // int min = (int) steiner->second;

    // printf("minimum: %d\n", min);
    // to_graphviz(tree, "result.dot");

    // destroy_graph(g);

    int V = atoi(argv[1]);
    int T = atoi(argv[2]);

    graph* g = make_randomly_connected_graph(V);

    set_t* t = make_set();

    for(int i=0; i<T; i++) {

        set_insert(t, rand() % (V + 1) );

    }
    
    pair* steiner = steiner_tree(g, t);

    destroy_graph((graph*) (steiner->first));
    free(steiner);

    destroy_set(t);
    destroy_graph(g);

    return 0;

}
