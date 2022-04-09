
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
     * @return graph* an empty graph.
     */
    graph* make_graph(int max_id);

    /**
     * @brief Makes a randomly connected graph with a maximum.
     * 
     * @param max_id the highest id that will be used to represent vertices in this graph.
     * @return graph* a randomly connected graph.
     */
    graph* make_randomly_connected_graph(int max_id);


    /**
     * @brief Inserts a new vertex in the graph.
     * If the id already exists in the graph nothing is done.
     * 
     * Best-Case: O(1)
     * Worst-Case: O(V)
     * Ammortized: O(1)
     * 
     * @param g the graph to insert a vertex in.
     * @param id the id of the vertex to insert.
     */
    void insert_vertex(graph* g, int id);

    /**
     * @brief Insert a new edge between two nodes with a given weight.
     * If an edge between the two vertices already exists nothing is done.
     * 
     * Complexity: O( deg(V) )
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
     * Complexity: O(max_id + E + V log(V))
     * 
     * @param g 
     * @param v1 
     * @param v2 
     * @return a pair containing first a graph pointer and second an integer.
     */
    pair* shortest_path(graph* g, int v1, int v2);

    /**
     * @brief Get the degree of the specified vertex.
     * 
     * @param g the graph to get to operate on.
     * @param id the id of the vertex.
     * @return int the degree of the vertex.
     */
    int degree(graph* g, int id);

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
     * @return int -1 if func never returned true of the id of the first vertex that returned true.
     */
    int dfs(graph* g, int start, int func(graph*, int, void*), void* input);

    /**
     * @brief Merges the graphs together on the specified node.
     * The objects inputed are destroyed in the process.
     * If w does not exist in one or more of the graphs the generated graph will not be connected.
     * 
     * O(max_id + V + E)
     * 
     * @param id The id of the node to merge the graphs.
     * @param ... comma delimited list of graphs terminated with a null pointer
     * @return graph* 
     */
    graph* graph_union(graph* g1, graph* g2);


    /**
     * @brief Prints to specified file the graphviz representation of the graph.
     * Complexity: O(E + V) writes.
     * 
     * @param g the graph to print.
     * @param filename the name of the file to print in.
     */
    void to_graphviz(graph* g, char* filename);

    /**
     * @brief Destroys and frees the graph.
     * 
     * @param g the graph to free.
     */
    void destroy_graph(graph* g);

#endif