/** 
 * \addtogroup APSP
 * @{ */

#ifndef SHORTESTPATH_H

    #define SHORTESTPATH_H

    #include "graph.h"
    #include "table.h"

    /**
     * @brief Calculates all pairs shortest path on the GPU and fills the result in the provided tables.
     *
     * @param [in] graph A pointer to a graph_t on the CPU.
     * @param [out] distances A pointer to a table_t to fill the respective distances in.
     * @param [out] predecessors A pointer to a table to fill the respective predecessors in.
     */
    __host__ void apsp_gpu_graph(graph::graph_t* graph, table::table_t* distances, table::table_t* predecessors);

#endif
/**@}*/
