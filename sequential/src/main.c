
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "graph.h"
#include "pair.h"
#include "steiner.h"
#include "set.h"

graph* test_graph() {

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

int main(int argc, char** argv) {
    
    graph* g = test_graph();

    to_graphviz(g, "test.dot");
    
    // pair* p1 = shortest_path(g, 1, 1);

    // graph* sp1   = (graph*) p1->first;
    // int    dist1 =    (int) p1->second;

    // to_graphviz(sp1, "sp1.dot");

    // pair* p2 = shortest_path(g, 1, 10);

    // graph* sp2   = (graph*) p2->first;
    // int    dist2 =    (int) p2->second;

    // to_graphviz(sp2, "sp2.dot");

    set_t* t = make_set();

    set_insert(t, 1);
    set_insert(t, 6);
    set_insert(t, 7);
    set_insert(t, 8);
    set_insert(t, 9);
    set_insert(t, 10);

    graph* steiner = steiner_tree(g, t, 2);
    
    to_graphviz(steiner, "st3.dot");

    destroy_graph(g);
    
    return 0;

}
