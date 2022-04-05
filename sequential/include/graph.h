
#ifndef GRAPH_H

    #define GRAPH_H

    typedef struct graph graph;

    /**
     * @brief Initializes a graph with V vertex.
     * 
     * @param V
     * @return graph* 
     */
    graph* make_graph(int V);

    /**
     * @brief Randomly connects all nodes randomly.
     * 
     * @param V an unconnected graph.
     */
    graph* make_randomly_connected_graph(int V);

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

    void destroy_graph(graph* g);

#endif