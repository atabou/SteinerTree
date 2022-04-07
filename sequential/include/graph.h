
#ifndef GRAPH_H

    #define GRAPH_H

    #include "pair.h"

    typedef struct graph graph;

    /**
     * @brief Creates an empty graph with a maximum id of max_id.
     * 
     * Complexity: O(max_id)
     * 
     * @param max_id the biggest possible id that will be entered in the graph.  
     * @return graph* An empty graph.
     */
    graph* make_graph(int max_id);

    /**
     * @brief Makes a randomly connected graph with a maximum.
     * 
     * @param max_id the highest id that will be used to represent vertices in this graph.
     */
    graph* make_randomly_connected_graph(int max_id);


    /**
     * @brief Inserts a new vertex in the graph (does not take into account collisions).
     * 
     * Best-Case: O(1)
     * Worst-Case: O(V)
     * Ammortized: O(1)
     * 
     * @param g 
     * @param id 
     * @return int 
     */
    void insert_vertex(graph* g, int id);

    /**
     * @brief Insert a new edge between two nodes with a given weight. (Does not check for duplicates)
     * 
     * Complexity: O(1)
     * 
     * @param g the graph to insert the edge in.
     * @param id1 id of the source node of the edge to insert.
     * @param id2 id of the destination node of the edge to insert.
     * @param w weight of the edge to insert. 
     */
    void insert_edge(graph* g, int id1, int id2, int w);

    /**
     * @brief Removes the vertex from the graph and clears all corresponding edges.
     * 
     * Complexity: O(V + E)
     * 
     * @param g a graph to delete a vertex from.
     * @param id the id of the vertex to delete.
     */
    void remove_vertex(graph* g, int id);

    /**
     * @brief Removes the edge between the specified ids.
     * If one of the specied ids does not exists nothing is done.
     * If the edge does not exist, nothing is done.
     * 
     * Complexity: O( deg(V) )
     * Best-Case: O(1)
     * Worst-Case: O(V - 1)
     * 
     * @param g the graph to delete an edge from.
     * @param id1 the source id of the vertex to delete.
     * @param id2 the destination of the vertex to delete.
     */
    void remove_edge(graph* g, int id1, int id2);

    /**
     * @brief Implementation of Dijkstra's shortest path algorithm with a fibonacci heap.
     * Complexity: O(E + V log(V))
     * 
     * @param g 
     * @param v1 
     * @param v2 
     * @return a pair containing first a graph pointer and second an integer.
     */
    pair* shortest_path(graph* g, int v1, int v2);

    int degree(graph* g, int id);

    int dfs(graph* g, int start, int func(graph*, int, void*), void* input);

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