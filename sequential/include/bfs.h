#ifndef BFS_H

    #define BFS_H

    #include "set.h"
    #include "graph.h"

    /**
     * @brief starts dfs of over the graph g from the specified node start.
     * Each time a node is encountered the specified function func is run.
     * If this function returns true, the id of the vertex at which it was invoked is returned, and dfs is stopped. 
     * The supplied function should take as input:
     *      - graph*: dfs will input g into this function
     *      - int: dfs will input the vertex it is currently processing.
     *      - void*: a void pointer that contains additional input data necessary to run func.
     * 
     * @param g the graph to run dfs on.
     * @param start the vertex to start dfs from.
     * @param func the search function to run at each vertex.
     * @param input the additional input for func.
     * @return set_t* the set of vertices that fulfill the conditions in the function func.
     */
    set_t* bfs(graph* g, int start, int func(graph*, int, void*), void* input);

#endif