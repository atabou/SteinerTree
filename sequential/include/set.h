
#ifndef SET_H

    #define SET_H

    typedef struct set_t set_t;

    set_t* make_set();

    void set_insert(set_t* set, int x);

    int get_element(set_t* X, int i);

    int set_size(set_t* X);

    int element_exists(int element, set_t* X);
    
    set_t* remove_element(int element, set_t* X);

    set_t* get_subset(set_t* X, long long mask);

    void print_set(set_t* X);

    void destroy_set(set_t* set);

#endif