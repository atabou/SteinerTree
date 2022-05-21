/** 
 * \addtogroup LinkedList
 * @{ */

#ifndef LLIST_H

    #define LLIST_H

    #include "inttypes.h"

    typedef struct llist_t llist_t;

    struct llist_t {

        int32_t dest; /** The destination vertex. */
        float weight; /** The weight of the edge. */
        llist_t*  next; /** The next edge in the linked list of edges. */

    };

    /**
     * @brief Adds a new node to the beginning of the linked list of edges.
     * 
     * @param [in] lst The linked list of edges to add to.
     * @param [in] v The destination of this node.
     * @param [in] w the weight of this node.
     * @return A pointer to the new beginning of the linked list of edges.
     */
    llist_t* llist_add(llist_t* lst, int32_t dest, float weight);

    /**
     * @brief Destroy and frees the linked list.
     * 
     * @param [in] lst A pointer to the head of the linked list of edges.
     */
    void destroy_llist(llist_t* lst);

#endif
/**@}*/
