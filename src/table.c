

#include <stdio.h>
#include <stdlib.h>

#include "table.h"

table_t* make_table(uint64_t n, uint64_t m) {

    table_t* t = (table_t*) malloc(sizeof(table_t));

    t->n = n;
    t->m = m;

    t->vals = (uint32_t*) malloc(sizeof(uint32_t) * n * m);

	return t;

}

void print_table(table_t* table) {
    
    printf("\n\033[0;32m   |");

    for(uint64_t i=0; i<table->m; i++) {
        printf("%2llu|", (unsigned long long) i);
    }

    printf("\n");
    for(uint64_t i=0; i<table->m; i++) {
        printf("+--");
    }
    printf("+--+\033[0m\n");

    for(uint64_t i=0; i<table->n; i++) {
        printf("\033[0;32m %2llu|\033[0m", (unsigned long long) i);
        for(uint64_t j=0; j<table->m; j++) {

            if(table->vals[i * table->m + j] == -1) {
                printf("\033[0;31m%2d\033[0m|", table->vals[i * table->m + j]);
            } else {
                printf("%2d|", table->vals[i * table->m + j]);
            }
            
        }
        printf("\n\033[0;32m+--+\033[0m");
        for(uint64_t j=0; j<table->m; j++) {
            printf("--+");
        }
        printf("\n");
    }
    printf("\n");

}

void free_table(table_t* t) {

    free(t->vals);

    t->vals = NULL;
    t->n = 0;
    t->m = 0;

    free(t);
    
}
