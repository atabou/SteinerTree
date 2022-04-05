
#ifndef GRAPH_H

    #define GRAPH_H

    #include "llist.h"

    typedef struct graph graph;

    struct graph {

        int     V;
        void**  data;
        int*    deg;
        llist** lst;

    };

    /**
     * @brief Implementation of Dijkstra's shortest path algorithm with a fibonacci heap.
     * Complexity: O(E + V log(V))
     * 
     * @param g 
     * @param v1 
     * @param v2 
     * @return int 
     */
    int shortest_path(graph* g, int v1, int v2);

    /**
     * @brief Prints to specified file the graphviz representation of the graph.
     * Complexity: O(E + V) writes.
     * 
     * @param g the graph to print.
     * @param filename the name of the file to print in.
     */
    void to_graphviz(graph* g, char* filename);

#endif