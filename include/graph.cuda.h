
#ifndef CUDAGRAPH_H

    #define CUDAGRAPH_H

//    extern "C" {
//        #include "graph.h"
//    }

    #include "graph.h"

    typedef graph_t cudagraph_t;

    cudagraph_t* copy_cudagraph(graph_t* cpu_graph);
    void free_cudagraph(cudagraph_t* g);


#endif
