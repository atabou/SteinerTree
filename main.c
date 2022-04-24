
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "graph.h"
#include "pair.h"
#include "steiner.h"
#include "set.h"

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

graph_t* test_graph2() {

    graph_t* g = make_graph();

    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);

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

void test() {
    
    graph_t* g = test_graph1();

    to_graphviz(g, "test.dot");
    
    set_t* t = make_set();

    // set_insert(t, 3);
    // set_insert(t, 5);

    set_insert(t, 0);
    set_insert(t, 5);
    set_insert(t, 6);
    set_insert(t, 7);
    set_insert(t, 8);
    set_insert(t, 9);
    
    table_t* steiner = steiner_tree(g, t);
    
    print_table(steiner);

    destroy_graph(g);
    destroy_set(t);
    free_table(steiner);

}

void specify_args(int argc, char** argv) {

    int V = atoi(argv[1]);
    int T = atoi(argv[2]);

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

void perf_test() {

    // clock_t c = clock();
    // for(uint64_t i=0; i<3llu * 3869835264llu; i++) {
    //     uint64_t l = i;
    // }
    // printf("Empty loop: %fs\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    for(int V=256; V<=1024; V*=2) {

        graph_t* g = make_randomly_connected_graph(V);

        for(int T=10; T<20; T++) {

            set_t* t = make_set();

            for(int i=0; i<T; i++) {

                set_insert(t, rand() % V );

            }

            table_t* steiner = steiner_tree(g, t);

            destroy_set(t);
            free_table(steiner);

        }

        
        destroy_graph(g);

    }

}

int main(int argc, char** argv) {
    
    test();
    // specify_args(argc, argv);
    // perf_test();

    return 0;
	
}
