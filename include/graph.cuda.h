/** 
 * \addtogroup GraphCUDA
 * @{ */

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
     * @param [in] graph A pointer to a graph_t on the CPU.
     */
    cudagraph_t* copy_cudagraph(graph_t* graph);
    
    /**
     * @brief Frees the given cudagraph_t from the gpu.
     *
     * @param [in] graph A pointer to a cudagraph_t on the GPU.
     */
    void free_cudagraph(cudagraph_t* graph);


#endif
/**@}*/
