
#ifndef LLIST_H

    #define LLIST_H

    #include "inttypes.h"

    typedef struct llist_t llist_t;

    struct llist_t {

        int32_t dest;
        float weight;
        llist_t*  next;

    };

    /**
     * @brief Adds a new node to the beginning of the linked list.
     * 
     * @param lst the list to add to.
     * @param v the data of this node.
     * @param w the weight of this node.
     * @return llist_t* a pointer to the new beginning of the list
     */
    llist_t* llist_add(llist_t* lst, int32_t dest, float weight);

    /**
     * @brief Destroy and frees the linked list
     * 
     * @param lst the linked list to destroy.
     * @param free_data a function that allows freeing the individual elements in the linked list.
     */
    void destroy_llist(llist_t* lst);

#endif
