
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

int load_gr_file(char* filename, graph** G, set_t** T) {

    FILE* fp = fopen(filename, "r");

    if(fp == NULL) {
        return -1;
    }

    *T = make_set();

    char* line = NULL;
    size_t buff = 0;
    ssize_t len = 0;

    int type = 0;

    while( (len = getline(&line, &buff, fp)) != -1) {

        char* token = strtok(line, " ");

        int type = 0;

        while(token != NULL) {

            if(type == 1) {

                int x = atoi(token);

                insert_vertex(*G, x);

                token = strtok(NULL, " ");
                int y = atoi(token);

                insert_vertex(*G, y);

                token = strtok(NULL, " ");
                int w = atoi(token);

                insert_edge(*G, x, y, w);
                insert_edge(*G, y, x, w);

            } else if(type == 2) {
                
                set_insert(*T, atoi(token));

            } else if(type == 3) {

                *G = make_graph(atoi(token));

            }

            if(token[0] == 'E' && token[1] == '\0') {
                type = 1;
            } else if(token[0] == 'T' && token[1] == '\0') {
                type = 2;
            } else if(token[0] == 'N' && token[1] == 'o') {
                type = 3;
            }

            token = strtok(NULL, " ");

        }

    }

    fclose(fp);

}

int next_combination(int n, int k, long long* mask) {

    if(*mask == 0) {

        *mask = (1ll << k) - 1ll;

    } else {

        long long c = (*mask) & -(*mask);
        long long r = (*mask) + c;
        *mask = r | (((r ^ (*mask)) >> 2)/c);

    }

    return *mask <= (1ll << n) - (1ll << (n-k));

}

void print_bits(long long number, int num_bits) {

    long long bit = 1ll << (num_bits - 1);

    while(bit != 0) {

        if(number & bit) {
            printf("1");
        } else{
            printf("0");
        }

        bit = bit >> 1;

    }


}

void print_table(int* table, int n, int m) {

    printf("\n\033[0;32m   |");

    for(int i=0; i<m; i++) {
        printf("%2d|", i);
    }

    printf("\n");
    for(int i=0; i<m; i++) {
        printf("+--");
    }
    printf("+--+\033[0m\n");

    for(int i=0; i<n; i++) {
        printf("\033[0;32m %2d|\033[0m", i);
        for(int j=0; j<m; j++) {

            if(table[i * m + j] == -1) {
                printf("\033[0;31m%2d\033[0m|", table[i * m + j]);
            } else {
                printf("%2d|", table[i * m + j]);
            }
            
        }
        printf("\n\033[0;32m+--+\033[0m");
        for(int j=0; j<m; j++) {
            printf("--+");
        }
        printf("\n");
    }
    printf("\n");

}

int max(int x, int y) {
    return (x > y) ? x : y;
}

int min(int x, int y) {
    return (x < y) ? x : y;
}