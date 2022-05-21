

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "table.h"

table_t* make_table(int32_t n, int32_t m) {

    table_t* t = (table_t*) malloc(sizeof(table_t));

    t->n = n;
    t->m = m;

    t->vals = (float*) malloc(sizeof(float) * n * m);

	return t;

}

void print_table(table_t* table) {
    
    printf("\n\033[0;32m    |");

    for(int32_t i=0; i<table->m; i++) {
        printf("%3d|", i);
    }

    printf("\n");
    for(int32_t i=0; i<table->m; i++) {
        printf("+---");
    }
    printf("+---+\033[0m\n");

    for(int32_t i=0; i<table->n; i++) {
        printf("\033[0;32m %3d|\033[0m", i);
        for(int32_t j=0; j<table->m; j++) {

            if(table->vals[i * table->m + j] == FLT_MAX) {
                printf("\033[0;31m%3d\033[0m|", -1);
            } else {
                printf("%.1f|", table->vals[i * table->m + j]);
            }
            
        }
        printf("\n\033[0;32m+---+\033[0m");
        for(int32_t j=0; j<table->m; j++) {
            printf("---+");
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
