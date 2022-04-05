
#ifndef LLIST_H

    #define LLIST_H

    typedef struct llist llist;

    struct llist {

        int     data;
        int     weight;
        llist*  next;

    };

    llist* llist_add(llist* lst, int v, int w);

    void destroy_llist(llist* lst);

#endif