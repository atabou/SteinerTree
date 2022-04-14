#ifndef SHORTESTPATH_H

    #define SHORTESTPATH_H

    #include "pair.h"
    #include "graph.h"

    /**
     * @brief returns shortest path algorithm with a fibonacci heap.
     * 
     * Complexity: O(max_id + E + V log(V))
     * 
     * @param g the graph to operate shortest path on.
     * @param v1 the source vertex.
     * @param v2 the destination vertex.
     * @return a pair with first a graph to pointer the path between v1 and v2, and second an integer representing the distance between v1 and v2.
     */
    pair* shortest_path(graph* g, int v1, int v2);

    /**
     * @brief 
     * 
     * @param g 
     * @return pair* 
     */
    pair* all_pairs_shortest_path(graph* g);

#endif