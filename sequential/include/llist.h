
#ifndef LLIST_H

    #define LLIST_H

    typedef struct llist llist;

    struct llist {

        int     data;
        int     weight;
        llist*  next;

    };

#endif