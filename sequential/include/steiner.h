
#ifndef STEINER_H

    #define STEINER_H

    #include "set.h"
    #include "graph.h"

    /**
     * @brief Calculates the steiner tree of the supplied graph ans set of terminals.
     * 
     * Complexity: O
     * 
     * @param g the graph to operate on.
     * @param terminals the set of terminals to calculate steiner tree.
     * @return graph* the steiner tree.
     */
    graph* steiner_tree(graph* g, set_t* terminals);

#endif