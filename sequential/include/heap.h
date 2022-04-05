
#ifndef HEAP_H

    #define HEAP_H

    typedef enum heap_type {

        FIBONACCI

    } heap_type;

    typedef struct heap heap;

    struct heap {

        void* data;

        void    (*insert)(heap* h, int n, int key);
        int     (*empty)(heap* h);
        int     (*find_min)(heap* h);
        int     (*extract_min)(heap* h);
        heap*   (*heap_union)(heap* h1, heap* h2);
        void    (*decrease_key)(heap* h, int n, int key);
        void    (*delete_node)(heap* h, int n);
        void    (*destroy)(heap* h);

    };

    heap* make_heap(heap_type type);

#endif