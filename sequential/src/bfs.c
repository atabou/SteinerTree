
#include "stdlib.h"

#include "bfs.h"

void _bfs(graph* g, int start, int* visited, int func(graph*, int, void*), void* input, set_t* result) {

    
    if(func(g, g->hash[start], input) == 1) {
        set_insert(result, g->hash[start]);
    }

    llist* curr = g->lst[start];

    while(curr != NULL) {

        int pos = (int) ((pair*) curr->data)->first;

        if(visited[pos] == 0) {

            visited[pos] = 1;
            _bfs(g, pos, visited, func, input, result);

        }

        curr = curr->next;

    }

}

set_t* bfs(graph* g, int start, int func(graph*, int, void*), void* input) {

    int visited[g->nVertices];

    for(int i=0; i<g->nVertices; i++) {
        visited[i] = 0;
    }

    set_t* result = make_set();

    _bfs(g, g->reverse_hash[start], visited, func, input, result);
    
    return result;

}