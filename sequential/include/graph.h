
#ifndef GRAPH_H

    #define GRAPH_H

    #include "pair.h"
    #include "set.h"
    #include "llist.h"

    typedef struct graph graph;

    struct graph {

        /**
        * Hash that links the internal ID of a vertex to a user specified ID.
        * If next_slot is less than nVertices then the slot represented by next_slot represents an empty "parent" slot.
        * 
        */
        int*    hash;

        int     nVertices; // The number of vertices in the graph. Also represents the leftmost empty position.
        int     capacity; // The capacity of the hash table.

        int*    reverse_hash; // Links the user specified IDs to its corresponding internal ID.
        int     max_id; // The biggest user specified ID. Also the size of the reverse hash table minus - 1.
        

        int*    deg; // Represents the degree of the node. If the degree of the node is -1 then the node does not exits.
        llist** lst; // The adjacency list of the graph.

    };

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
     * @brief Returns the number of vertices in the graph.
     * 
     * Complexity: O(1)
     * 
     * @param g the graph the get the number of vertices from.
     * @return int 
     */
    int number_of_vertices(graph* g);

    /**
     * @brief Returns the number of edges in the graph.
     * 
     * Complexity: O(V)
     * 
     * @param g the graph the get the number of edges from.
     * @return int the number of vertices int the graph.
     */
    int number_of_edges(graph* g);

    /**
     * @brief Get the degree of the specified vertex.
     * 
     * @param g the graph to get to operate on.
     * @param id the id of the vertex.
     * @return int the degree of the vertex.
     */
    int degree(graph* g, int id);

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