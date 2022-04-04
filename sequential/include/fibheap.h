
#ifndef FIBHEAP_H

    #define FIBHEAP_H

    typedef struct fibheap fibheap;

    fibheap*    make_fibheap();
    void        fibheap_insert(fibheap* fib, int n, int key);
    int         fibheap_find_min(fibheap* fib);
    int         fibheap_extract_min(fibheap* fib);
    fibheap*    fibheap_union(fibheap* fib1, fibheap* fib2);
    int         fibheap_decrease_key(fibheap* fib, int n, int key);
    int         fibheap_delete(fibheap* fib, int n);
    void        destroy_fibheap(fibheap* fib);

#endif