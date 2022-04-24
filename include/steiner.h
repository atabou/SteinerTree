
#ifndef STEINER_H

    #define STEINER_H

    #include "set.h"
    #include "graph.h"
    #include "table.h"
    #include "pair.h"

    table_t* steiner_tree(graph_t* g, set_t* t);
    void fill_steiner_dp_table_cpu(table_t* table, graph_t* g, set_t* t, table_t* distances);
    void fill_steiner_dp_table_gpu(table_t* table, graph_t* g, set_t* t, table_t* distances);


#endif
