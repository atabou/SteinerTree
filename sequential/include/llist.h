
#ifndef LLIST_H

    #define LLIST_H

    typedef struct llist llist;

    struct llist {

        int     data;
        int     weight;
        llist*  next;

    };

    /**
     * @brief Adds a new node to the beginning of the linked list.
     * 
     * @param lst the list to add to.
     * @param v the data of this node.
     * @param w the weight of this node.
     * @return llist* a pointer to the new beginning of the list
     */
    llist* llist_add(llist* lst, int v, int w);

    /**
     * @brief Destroy and frees the linked list
     * 
     * @param lst the linked list to destroy.
     */
    void destroy_llist(llist* lst);

#endif