
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "graph.h"
#include "pair.h"
#include "steiner.h"
#include "set.h"

void load_gr_file(char* filename, graph_t** g, set_t** t, uint32_t** h, uint32_t* hsize) {

	FILE* fp = fopen(filename, "r");

	if(fp == NULL) {

		*g = NULL;
		*t = NULL;
		*h = NULL;
		return;
	
	}

	*t = make_set();
	*g = make_graph();
	
	*h = (uint32_t*) malloc(sizeof(uint32_t));
	(*h)[0] = UINT32_MAX;
	*hsize = 1;

	char* line = NULL;
	size_t buff = 0;
	ssize_t len = 0;

	while( (len  = getline(&line, &buff, fp)) != -1 ) {
	
		char* token = strtok(line, " ");
		int type = 0;

		while(token != NULL) {

			if(type == 1) {

				uint32_t x = atoi(token);

				if(x >= *hsize) {

					*h = (uint32_t*) realloc(*h, sizeof(uint32_t) * (x + 1));
					
					for(uint32_t i=*hsize; i < (x+1); i++) {
						(*h)[i] = UINT32_MAX;
					}

					*hsize = x + 1;

				}

				if( (*h)[x] == UINT32_MAX ) {

					uint32_t id = insert_vertex(*g);
					(*h)[x] = id; 

				}

				token = strtok(NULL, " ");
				uint32_t y = atoi(token);

				if(y >= *hsize) {

					*h = (uint32_t*) realloc(*h, sizeof(uint32_t) * (y + 1));
					
					for(uint32_t i=*hsize; i < (y+1); i++) {
						(*h)[i] = UINT32_MAX;
					}

					*hsize = y + 1;

				}

				if( (*h)[y] == UINT32_MAX ) {

					uint32_t id = insert_vertex(*g);
					(*h)[y] = id; 

				}

				token = strtok(NULL, " ");
				uint32_t w = atoi(token);

				insert_edge(*g, (*h)[x], (*h)[y], w);
				insert_edge(*g, (*h)[y], (*h)[x], w);

			} else if(type == 2) {

				uint32_t val = atoi(token);
				set_insert(*t, (*h)[val]);

			}

			if(token[0] == 'E' && token[1] == '\0') {
				type = 1;
			} else if(token[0] == 'T' && token[1] == '\0') {
				type = 2;
			}

			token = strtok(NULL, " ");

		}

	}

	fclose(fp);

}

graph_t* test_graph1() {

    graph_t* g = make_graph();

    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);

    insert_edge(g, 0, 1, 1);
    insert_edge(g, 0, 1, 1);
    insert_edge(g, 0, 2, 1);
    insert_edge(g, 0, 3, 1);
    insert_edge(g, 0, 4, 1);
    insert_edge(g, 1, 5, 1);
    insert_edge(g, 1, 6, 1);
    insert_edge(g, 1, 8, 1);
    insert_edge(g, 2, 5, 1);
    insert_edge(g, 2, 7, 1);
    insert_edge(g, 3, 7, 1);
    insert_edge(g, 3, 9, 1);
    insert_edge(g, 4, 6, 1);
    insert_edge(g, 4, 9, 1);

    insert_edge(g, 1, 0, 1);
    insert_edge(g, 1, 0, 1);
    insert_edge(g, 2, 0, 1);
    insert_edge(g, 3, 0, 1);
    insert_edge(g, 4, 0, 1);
    insert_edge(g, 5, 1, 1);
    insert_edge(g, 6, 1, 1);
    insert_edge(g, 8, 1, 1);
    insert_edge(g, 5, 2, 1);
    insert_edge(g, 7, 2, 1);
    insert_edge(g, 7, 3, 1);
    insert_edge(g, 9, 3, 1);
    insert_edge(g, 6, 4, 1);
    insert_edge(g, 9, 4, 1);

    return g;

}

graph_t* test_graph2() {

    graph_t* g = make_graph();

    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);
    insert_vertex(g);

    insert_edge(g, 0, 1, 1);
    insert_edge(g, 1, 0, 1);

    insert_edge(g, 1, 2, 1);
    insert_edge(g, 2, 1, 1);

    insert_edge(g, 2, 3, 1);
    insert_edge(g, 3, 2, 1);

    insert_edge(g, 1, 4, 1);
    insert_edge(g, 4, 1, 1);

    insert_edge(g, 2, 4, 1);
    insert_edge(g, 4, 2, 1);

    insert_edge(g, 4, 5, 1);
    insert_edge(g, 5, 4, 1);

    insert_edge(g, 4, 6, 1);
    insert_edge(g, 6, 4, 1);

    return g;

}

graph_t* make_randomly_connected_graph(uint32_t v) {

    graph_t* g = make_graph();

    for(int i=0; i<v; i++) {
        insert_vertex(g);
    }

    int used[v];

    for(int i=0; i<v; i++) {

        for(int j=0; j<v; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<v; j++) { // Shuffle array

            int x = rand() % v;
            int y = rand() % v;

            int tmp = used[x];
            used[x] = used[y];
            used[y] = tmp;

        }

        int deg = rand() % v;

        for(int j=0; j<deg; j++) {

            if(used[j] != -1) {

                insert_edge(g, i, used[j], 1);
                insert_edge(g, used[j], i, 1);

            }

        }

    }

    return g;

}

void test() {
    
    graph_t* g = test_graph1();

    to_graphviz(g, "test.dot");
    
    set_t* t = make_set();

    // set_insert(t, 3);
    // set_insert(t, 5);

    set_insert(t, 0);
    set_insert(t, 5);
    set_insert(t, 6);
    set_insert(t, 7);
    set_insert(t, 8);
    set_insert(t, 9);
    
    table_t* steiner = steiner_tree(g, t);
    
    print_table(steiner);

    destroy_graph(g);
    destroy_set(t);
    free_table(steiner);

}

void specify_args(int argc, char** argv) {

    uint32_t V = atoi(argv[1]);
    uint32_t T = atoi(argv[2]);

    graph_t* g = make_randomly_connected_graph(V);

    set_t* t = make_set();

    for(int i=0; i<T; i++) {

        set_insert(t, rand() % (V + 1) );

    }
    
    table_t* steiner = steiner_tree(g, t);
    
    destroy_set(t);
    destroy_graph(g);
    free_table(steiner);

}

void perf_test() {

    // clock_t c = clock();
    // for(uint64_t i=0; i<3llu * 3869835264llu; i++) {
    //     uint64_t l = i;
    // }
    // printf("Empty loop: %fs\n", (double) (clock() - c) / CLOCKS_PER_SEC);

    for(int V=64; V<=1024; V*=2) {

        graph_t* g = make_randomly_connected_graph(V);

        for(int T=2; T<10; T++) {

            set_t* t = make_set();

            for(int i=0; i<T; i++) {

                set_insert(t, rand() % V );

            }

            table_t* steiner = steiner_tree(g, t);

            destroy_set(t);
            free_table(steiner);

        }

        
        destroy_graph(g);

    }

}

void gr_file_test() {

	graph_t* g = NULL;
	set_t* t = NULL;

	uint32_t* h = NULL;
	uint32_t hsize = 0;


	for(int i=1; i<=50; i+=2) {

		char str[100];
		sprintf(str, "tests/instance%03d.gr", i);

		load_gr_file(str, &g, &t, &h, &hsize);	

		printf("%s - ", str);
		table_t* result = steiner_tree(g, t);

		destroy_graph(g);
		destroy_set(t);
		free(h);

		free(result);


	}
	
}

int main(int argc, char** argv) {
    
    // test();
    specify_args(argc, argv);
    // perf_test();
	/* gr_file_test(); */


    return 0;
	
}
