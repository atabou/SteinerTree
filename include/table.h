
#ifndef TABLE_H

    #define TABLE_H

    #include <stdint.h>

    typedef struct table_t table_t;

    struct table_t {

        uint32_t* vals;

        uint64_t  n;
        uint64_t  m;

    };

    table_t* make_table(uint64_t n, uint64_t m);
    void print_table(table_t* table);
    void free_table(table_t* t);

#endif