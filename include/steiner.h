
#ifndef STEINER_H

    #define STEINER_H

    #include "set.h"
    #include "graph.h"
    #include "table.h"

    float steiner_tree(graph_t* graph, set_t* terminals, table_t* distances);

#endif
