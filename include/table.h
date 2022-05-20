
#ifndef TABLE_H

    #define TABLE_H

    #include <stdint.h>

    typedef struct table_t table_t;

    struct table_t {

        float* vals;

        int32_t  n;
        int32_t  m;

    };

    table_t* make_table(int32_t n, int32_t m);
    void print_table(table_t* table);
    void free_table(table_t* t);

#endif
