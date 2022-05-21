
#ifndef STEINER_H

    #define STEINER_H

    #include "set.h"
    #include "graph.h"
    #include "table.h"

    table_t* steiner_tree(graph_t* graph, set_t* terminals, table_t* distances);

#endif
