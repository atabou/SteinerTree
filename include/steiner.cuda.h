/** 
 * \addtogroup SteinerGPU
 * @{ */

#ifndef STEINER2_CUDA_H

    #define STEINER2_CUDA_H

    #include "set.cuda.h"
    #include "graph.cuda.h"
    #include "table.cuda.h"

    /**
     * @brief Calculates the minimum steiner tree of the provided graph.
     *
     * @param [in] g A pointer to a cudagraph_t on the GPU to calculate the minimum steiner tree for.
     * @param [in] g_size The number of vertices in the cudagraph_t g.
     * @param [in] t A pointer to a cudaset_t on the GPU representing the number of terminal vertices.
     * @param [in] t_size The number of terminals in the cudaset_t t.
     * @param [in] distances A pointer to a cudatable_t containing the all pairs shortest path of the cudagraph_t g/
     */
     float steiner_tree_gpu(cudagraph_t* g, int32_t g_size, cudaset_t* t, int32_t t_size, cudatable_t* distances);


#endif
/**@}*/
