#ifndef SHORTESTPATH_H

    #define SHORTESTPATH_H

    #include "pair.h"
    #include "graph.h"

    /**
     * @brief Implementation of Dijkstra's shortest path algorithm with a fibonacci heap.
     * Complexity: O(max_id + E + V log(V))
     * 
     * @param g 
     * @param v1 
     * @param v2 
     * @return a pair containing first a graph pointer and second an integer.
     */
    pair* shortest_path(graph* g, int v1, int v2);

    pair* all_pairs_shortest_path(graph* g);

#endif