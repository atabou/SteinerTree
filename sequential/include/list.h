
#ifndef LLIST_H

    #define LLIST_H

    typedef struct llist llist;

    struct llist {

        int     data;
        llist*  next;

    };

    typedef struct dlist dlist;

    struct dlist {

        int     data;
        dlist*  next;
        dlist*  prev;

    };

#endif