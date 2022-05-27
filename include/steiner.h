/** 
 * \addtogroup SteinerGPU
 * @{ */

#ifndef STEINER_H

    #define STEINER_H

    #include "set.h"
    #include "graph.h"
    #include "table.h"
    
    typedef struct steiner_result {

        graph_t* mst; /** The minimum steiner tree extracted. */
        float cost; /** The cost of the minimum steiner tree. */

    } steiner_result;


    /**
     * @brief Calculates the minimum steiner tree of the provided graph on the CPU.
     *
     * @param [in] graph A pointer to a graph_t on the CPU to calculate the minimum steiner tree for.
     * @param [in] terminals A pointer to a set_t on the CPU representing the number of terminal vertices.
     * @param [in] distances A pointer to a table_t containing the all pairs shortest path of the provided graph_t
     */
    steiner_result steiner_tree_cpu(graph_t* graph, set_t* terminals, table_t* distances);

    /**
     * @brief Calculates the minimum steiner tree of the provided graph on the GPU.
     *
     * @param [in] graph A pointer to a cudagraph_t on the GPU to calculate the minimum steiner tree for.
     * @param [in] graph_size The number of vertices in the cudagraph_t g.
     * @param [in] terminals A pointer to a cudaset_t on the GPU representing the number of terminal vertices.
     * @param [in] terminals_size The number of terminals in the cudaset_t t.
     * @param [in] distances A pointer to a cudatable_t containing the all pairs shortest path of the supplied cudagraph_t on the GPU.
     */
    steiner_result steiner_tree_gpu( cudagraph_t* graph, 
                            int32_t      graph_size, 
                            cudaset_t*   terminals, 
                            int32_t      terminals_size, 
                            cudatable_t* distances );




#endif
/**@}*/
