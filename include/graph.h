/** 
 * \addtogroup Graph 
 * @{ */
#ifndef GRAPH_H

    #define GRAPH_H

    #include <stdint.h>

    typedef struct graph_t {

        int32_t   max; /** Current capacity of the graph. */
        int32_t   vrt; /** Number of vertices in the graph. */
        int32_t*  deg; /** Represents the degree of the node. If the degree of the node is -1 then the node does not exist. */
        int32_t** dst; /** 2D array that contains the destination vertices of the edges. */
        float** wgt; /** 2D array that contains the weights of the edges */

    } graph_t;

    /**
     * @brief Creates an empty graph_t.
     *  
     * @return A pointer to an empty graph_t.
     */
    graph_t* make_graph();

    /**
     * @brief Inserts a new vertex in the graph_t.
     * 
     * Best-Case: O(1)
     * Worst-Case: O(V)
     * Ammortized: O(1)
     * 
     * @param [in] g a pointer to the graph to insert a vertex in.
     * @return The id of the vertex that was inserted.
     */
    uint32_t insert_vertex(graph_t* g);

    /**
     * @brief Insert a new edge between two nodes with a given weight.
     * If an edge between the two vertices already exists nothing is done.
     * 
     * Complexity: O( deg(V) )
     * 
     * @param [in] g A pointer to the graph_t to insert the edge in.
     * @param [in] id1 The id of the source node of the edge to insert.
     * @param [in] id2 The id of the destination node of the edge to insert.
     * @param [in] w weight of the edge to insert. 
     */
    void insert_edge(graph_t* g, int32_t id1, int32_t id2, float w);

    void to_graphviz(graph_t* g, char* filename);

    /**
     * @brief Destroys and frees the graph_t.
     * 
     * @param [in] g a pointer to the graph_t to destroy.
     */
    void destroy_graph(graph_t* g);


#endif

/** @} */
