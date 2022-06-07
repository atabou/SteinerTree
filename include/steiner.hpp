/** 
 * \addtogroup SteinerGPU
 * @{ */

#ifndef STEINER_H

    #define STEINER_H

    #include "query.hpp"
    #include "graph.hpp"
    #include "table.hpp"
    #include "tree.hpp"

    namespace steiner {

        struct result_t {

            table::table_t< float >* costs;
            table::table_t<int32_t>* roots;
            table::table_t<int64_t>* trees;

            float   cost; /** The cost of the minimum steiner tree. */
            int32_t root;
            int64_t tree;

            tree::tree_t* mst;

            graph::graph_t* subgraph; /** The minimum steiner tree extracted. */

        };


        void fill_cpu(graph::graph_t* graph, query::query_t* terminals, table::table_t<float> distances, result_t** result);
        void fill_gpu(cudagraph::graph_t* graph, cudaquery::query_t* terminals, cudatable::table_t<float>* distances, result_t** result);
        void backtrack(query::query_t* terminals, result_t* result);
        void branch_and_clean(table::table_t<int32_t>* predecessors, result_t* result);
        void build_subgraph(graph::graph_t* graph, result_t* result);
 
        void destroy(result_t* result);

    };
    
    /**
     * @brief Calculates the minimum steiner tree of the provided graph on the CPU.
     *
     * @param [in] graph A pointer to a graph_t on the CPU to calculate the minimum steiner tree for.
     * @param [in] terminals A pointer to a set_t on the CPU representing the number of terminal vertices.
     * @param [in] distances A pointer to a table_t containing the all pairs shortest path of the provided graph_t
     * @param [out] result The address of a pointer to the result object of the steiner tree computation.
     */
    void steiner_tree_cpu(graph::graph_t* graph, query::query_t* terminals, table::table_t<float>* distances, steiner::result_t** result);

    /**
     * @brief Calculates the minimum steiner tree of the provided graph on the GPU.
     *
     * @param [in] graph_d A pointer to a cudagraph_t on the GPU to calculate the minimum steiner tree for.
     * @param [in] graph_size The number of vertices in the cudagraph_t g.
     * @param [in] terminals_d A pointer to a cudaset_t on the GPU representing the number of terminal vertices.
     * @param [in] terminals_size The number of terminals in the cudaset_t t.
     * @param [in] distances A pointer to a cudatable_t containing the all pairs shortest path of the supplied cudagraph_t on the GPU.
     * @param [out] result The address of a pointer to the result object of the steiner tree computation.
     */
    void steiner_tree_gpu( cudagraph::graph_t*        graph_d, 
                int32_t                    graph_size, 
                cudaquery::query_t*        terminals_d, 
                int32_t                    terminals_size, 
                cudatable::table_t<float>* distances,
                table::table_t<int32_t>*   predecessors,
                steiner::result_t**                 result );


#endif
/**@}*/
