
#ifndef CUDAGRAPH_H

    #define CUDAGRAPH_H

    #include "graph.h"

    /**
     * @brief Redefinition of graph_t to make it clear that a given graph_t is stored on the GPU.
     */
    typedef graph_t cudagraph_t;

    /**
     * @brief Copies the given graph on the GPU.
     *
     * @param [in] graph_t a pointer to a non-empty graph_t.
     */
    cudagraph_t* copy_cudagraph(graph_t* cpu_graph);
    
    /**
     * @brief Frees the given cudagraph_t from the gpu.
     *
     * @param [in] cudagraph_t A pointer to a cudagraph_t on the GPU.
     */
    void free_cudagraph(cudagraph_t* g);


#endif
