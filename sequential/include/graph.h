
#ifndef GRAPH_H

    #define GRAPH_H

    #include "llist.h"

    typedef struct graph graph;

    struct graph {

        int     V;
        void**  data;
        int*    deg;
        llist** lst;

    };

    int shortest_path(graph* g, int v1, int v2);

#endif