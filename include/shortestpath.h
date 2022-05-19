#ifndef SHORTESTPATH_H

    #define SHORTESTPATH_H

    #include "graph.h"
    #include "table.h"

    void apsp_gpu_graph(graph_t* graph, table_t* distances, table_t* predecessors);

#endif
