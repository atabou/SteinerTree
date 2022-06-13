/** 
 * \addtogroup Graph 
 * @{ */
#ifndef GRAPH_H

    #define GRAPH_H

    #include <stdint.h>


    namespace graph {


        typedef struct graph_t {

            int32_t   max; /** Current capacity of the graph. */
            int32_t   vrt; /** Number of vertices in the graph. */
            int32_t*  deg; /** Represents the degree of the node. If the degree of the node is -1 then the node does not exist. */
            int32_t** dst; /** 2D array that contains the destination vertices of the edges. */
            float**   wgt; /** 2D array that contains the weights of the edges */

        } graph_t;


        /**
         * @brief Creates an empty graph_t.
         *  
         * @param [out] graph A pointer to a graph_t pointer to initialize.
         */
         void make(graph_t** graph);


        /**
         * @brief Inserts a new vertex in the graph_t.
         * 
         * Best-Case: O(1)
         * Worst-Case: O(V)
         * Ammortized: O(1)
         * 
         * @param [in] graph A pointer to the graph to insert a vertex in.
         * @return The id of the vertex that was inserted.
         */
         int32_t insert_vertex(graph_t* graph);


        /**
         * @brief Insert a new edge between two nodes with a given weight.
         * If an edge between the two vertices already exists nothing is done.
         * 
         * Complexity: O( deg(V) )
         * 
         * @param [in] graph A pointer to the graph_t to insert the edge in.
         * @param [in] src The id of the source node of the edge to insert.
         * @param [in] dst The id of the destination node of the edge to insert.
         * @param [in] wgt The weight of the edge to insert. 
         */
         void insert_edge(graph_t* graph, int32_t src, int32_t dst, float wgt);

         float weight(graph_t* graph, int32_t src, int32_t dst);

        /** 
         * @brief Prints the specified graph_t to a file in the graphviz format.
         * 
         * @param [in] graph The graph to print.
         * @param [in] graph The file name to output the graphviz representation to.
         */
         void to_graphviz(graph_t* graph, char* filename);


        /**
         * @brief Destroys and frees the graph_t.
         * 
         * @param [in] graph a pointer to the graph_t to destroy.
         */
         void destroy(graph_t* graph);


    }


    namespace cudagraph {

        struct graph_t {

            graph::graph_t* graph;
            int32_t vrt;

        };
        
        /**
         * @brief Copies the given graph on the GPU.
         *
         * @param [in] graph A pointer to a graph_t on the CPU.
         */
         void transfer_to_gpu(graph_t** graph_d, graph::graph_t* graph);
        

        /**
         * @brief Frees the given cudagraph_t from the gpu.
         *
         * @param [in] graph A pointer to a cudagraph_t on the GPU.
         */
         void destroy(graph_t* graph);

    }


#endif

/** @} */
