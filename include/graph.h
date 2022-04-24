
#ifndef GRAPH_H

    #define GRAPH_H

    #include <stdint.h>
    
    #include "llist.h"

    typedef struct graph_t graph_t;

    struct graph_t {

        uint32_t  max; // Current capacity of the graph.
        uint32_t  vrt; // Number of vertices in the graph.
        uint32_t* deg; // Represents the degree of the node. If the degree of the node is -1 then the node does not exits.
        llist_t** lst; // The adjacency list of the graph_t.

    };

    /**
     * @brief Creates an empty graph_t with a maximum id of max_id.
     * 
     * Complexity: O(max_id)
     * 
     * @param max_id the biggest possible id that will be entered in the graph_t.  
     * @return graph_t* an empty graph_t.
     */
    graph_t* make_graph();

    /**
     * @brief Inserts a new vertex in the graph_t.
     * 
     * Best-Case: O(1)
     * Worst-Case: O(V)
     * Ammortized: O(1)
     * 
     * @param g the graph_t to insert a vertex in.
     * @return uint32* the id of the vertex that was inserted.
     */
    uint32_t insert_vertex(graph_t* g);

    /**
     * @brief Insert a new edge between two nodes with a given weight.
     * If an edge between the two vertices already exists nothing is done.
     * 
     * Complexity: O( deg(V) )
     * 
     * @param g the graph_t to insert the edge in.
     * @param id1 id of the source node of the edge to insert.
     * @param id2 id of the destination node of the edge to insert.
     * @param w weight of the edge to insert. 
     */
    void insert_edge(graph_t* g, uint32_t id1, uint32_t id2, uint32_t w);

    void to_graphviz(graph_t* g, char* filename);

    /**
     * @brief Destroys and frees the graph_t.
     * 
     * @param g the graph_t to free.
     */
    void destroy_graph(graph_t* g);


#endif