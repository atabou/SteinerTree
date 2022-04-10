
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "llist.h"
#include "graph.h"
#include "heap.h"
#include "pair.h"
#include "set.h"

struct graph {

    /**
    * Hash that links the internal ID of a vertex to a user specified ID.
    * If next_slot is less than nVertices then the slot represented by next_slot represents an empty "parent" slot.
    * 
    */
    int*    hash;

    int     nVertices; // The number of vertices in the graph. Also represents the leftmost empty position.
    int     capacity; // The capacity of the hash table.

    int*    reverse_hash; // Links the user specified IDs to its corresponding internal ID.
    int     max_id; // The biggest user specified ID. Also the size of the reverse hash table minus - 1.
    

    int*    deg; // Represents the degree of the node. If the degree of the node is -1 then the node does not exits.
    llist** lst; // The adjacency list of the graph.

};


graph* make_graph(int max_id) {

    graph* g = (graph*) malloc(sizeof(graph));

    g->capacity     = 0;
    g->nVertices    = 0;
    g->max_id       = max_id + 1;

    g->hash         = NULL;
    g->deg          = NULL;
    g->lst          = NULL;

    g->reverse_hash = (int*) malloc(sizeof(graph) * (max_id + 1));

    for(int i=0; i<max_id + 1; i++) {
        g->reverse_hash[i] = -1;
    }    

    return g;

}

void insert_vertex(graph* g, int id) {

    if(id >= 0 && id < g->max_id && g->reverse_hash[id] == -1) {

        if(g->capacity == 0) {

            g->hash = (int*) malloc( sizeof(int) );
            g->deg  = (int*) malloc( sizeof(int) );
            g->lst  = (llist**) malloc(sizeof(llist*));
            
            g->capacity = 1;

        } else if(g->nVertices >= g->capacity) { // O(V) normally, O(1) ammortized.

            g->hash = (int*) realloc(g->hash, sizeof(int) * 2 * g->capacity);
            g->deg  = (int*) realloc(g->deg, sizeof(int) * 2 * g->capacity);
            g->lst  = (llist**) realloc(g->lst, sizeof(llist*) * 2 * g->capacity);

            g->capacity = 2 * g->capacity;

        }

        g->hash[g->nVertices] = id;
        g->reverse_hash[id]   = g->nVertices;
        g->deg[g->nVertices]  = 0;
        g->lst[g->nVertices]  = NULL;

        g->nVertices = g->nVertices + 1;

    }

}

void insert_edge(graph* g, int id1, int id2, int w) {

    if(id1 >= 0 && id2 >= 0 && id1 < g->max_id && id2 < g->max_id) {

        int internal1 = g->reverse_hash[id1];
        int internal2 = g->reverse_hash[id2];

        if(internal1 != -1 && internal2 != -1) {

            llist* e = g->lst[internal1];

            while(e != NULL) {

                if(e->data == internal2) {
                    return;
                }

                e = e->next;

            }

            g->lst[internal1] = llist_add(g->lst[internal1], internal2, w);
            g->deg[internal1]++;

        }

    }

}

void remove_vertex(graph* g, int id) {

    if(id > 0 && id < g->max_id) {

        int internal = g->reverse_hash[id];

        if(internal != -1) {

            // Removes all edges related to id.

            for(int i=0; i<g->nVertices; i++) { // O(E)

                if(i == internal) {

                    destroy_llist(g->lst[i]);
                    g->lst[i] = NULL;
                    g->deg[i] = 0;

                } else {

                    remove_edge(g, i, internal);

                }

            }

            // Move all upstream vertices and their ids down.

            for(int i=internal; i<g->nVertices - 1; i++) { // O(V)

                g->reverse_hash[g->hash[i+1]] = i;
                g->hash[i] = g->hash[i+1];
                g->deg[i] = g->deg[i+1];
                g->lst[i] = g->lst[i+1];

            }

            // Clear the corresponding reverse hash

            g->reverse_hash[id] = -1;

            // Set the last position in the hash as availble
            
            g->hash[g->nVertices - 1] = -1;
            g->deg[g->nVertices - 1]  = 0;
            g->lst[g->nVertices - 1]  = NULL;
            g->nVertices--;
            
        }

    }

}

void remove_edge(graph* g, int id1, int id2) {

    if(id1 >= 0 && id2 >= 0 && id1 < g->max_id && id2 < g->max_id) {

        int internal1 = g->reverse_hash[id1];
        int internal2 = g->reverse_hash[id2];

        if(g->lst[internal1] != NULL) {

            if(g->lst[internal1]->data == internal2) {

                llist* del = g->lst[internal1];
                g->lst[internal1] = g->lst[internal1]->next;
                del->next = NULL;

                destroy_llist(del);
                del = NULL;

                g->deg[internal1]--;

            } else {

                llist* curr = g->lst[internal1];

                while(curr->next != NULL && curr->next->data != internal2) {
                    curr = curr->next;
                }

                if(curr->next != NULL) {

                    llist* del = curr->next;
                    curr->next = curr->next->next;
                    del->next = NULL;
                    
                    destroy_llist(del);
                    del = NULL;

                    g->deg[internal1]--;

                }

            }

        }

    }


}

graph* make_randomly_connected_graph(int max_id) {

    graph* g = make_graph(max_id);

    for(int i=0; i<g->max_id + 1; i++) {
        insert_vertex(g, i);
    }

    int used[max_id + 1];

    for(int i=0; i<max_id+1; i++) {

        for(int j=0; j<max_id + 1; j++) {
            used[j] = j;
        }

        used[i] = -1;

        for(int j=0; j<max_id + 1; j++) { // Shuffle array

            int x = rand() % (max_id + 1);
            int y = rand() % (max_id + 1);

            int tmp = used[x];
            used[x] = used[y];
            used[y] = tmp;

        }

        int deg = rand() % max_id;

        for(int j=0; j<deg; j++) {

            if(used[j] != -1) {

                printf("(%d, %d)\n", i, used[j]);
                insert_edge(g, i, used[j], 1);

            }

        }

    }

    return g;

}

int degree(graph* g, int id) {

    if(id >= 0 && id < g->max_id) {

        int k = g->reverse_hash[id];

        if(k != -1) {

            return g->deg[k];

        } else {

            return -1;

        }

    }

    

}

void _dfs_rec(graph* g, int start, int* visited, int func(graph*, int, void*), void* input, int* result) {

    if(func(g, g->hash[start], input) == 1) {
        *result = start;
        return;
    }

    llist* curr = g->lst[start];

    while(curr != NULL && *result == -1) {

        if(visited[curr->data] == 0) {

            visited[curr->data] = 1;
            _dfs_rec(g, curr->data, visited, func, input, result);

        }

        curr = curr->next;

    }

}

int _dfs(graph* g, int start, int func(graph*, int, void*), void* input) {

    int visited[g->nVertices];

    for(int i=0; i<g->nVertices; i++) {
        visited[i] = 0;
    }

    int result = -1;

    _dfs_rec(g, start, visited, func, input, &result);

    return result;

}

int dfs(graph* g, int start, int func(graph*, int, void*), void* input) {

    return g->reverse_hash[_dfs(g, g->reverse_hash[start], func, input)];

}

pair* dijkstra(graph* g, int internal1) {

    // Initialize Single Source Shortest Path

    int* distances = (int*) malloc(sizeof(int) * g->nVertices);
    int* parents   = (int*) malloc(sizeof(int) * g->nVertices);

    for(int i=0; i<g->nVertices; i++) {

        distances[i] = INT_MAX;
        parents[i] = -1;

    }

    distances[internal1] = 0;

    // Initialize minheap.

    heap* pq = make_heap(FIBONACCI, g->nVertices);

    for(int i=0; i < g->nVertices; i++) {
        pq->insert(pq, i, distances[i]);
    }

    // Calculates shortest path

    while( !pq->empty(pq) ) {

        int u = pq->extract_min(pq);

        llist* curr = g->lst[u];

        while(curr != NULL) {

            int v = curr->data;

            if(distances[v] > distances[u] + curr->weight) {

                distances[v] = distances[u] + curr->weight;
                pq->decrease_key(pq, v, distances[v]);
                parents[v] = u;

            }

            curr = curr->next;

        }

    }

    pq->destroy(pq); // O(1) because the priority q is empty at this point.

    return make_pair(distances, parents);

}

pair* shortest_path(graph* g, int v1, int v2) {

    // Check boundary conditions

    if(v1 < 0 || v2 < 0 || v1 >= g->max_id || v2 >= g->max_id) {
        return NULL;
    }

    int internal1 = g->reverse_hash[v1];
    int internal2 = g->reverse_hash[v2];

    if(internal1 == -1 || internal2 == -1) {
        return NULL;
    }

    pair* p = dijkstra(g, internal1);

    int* distances = (int*) p->first;
    int* parents = (int*) p->second;

    graph* path = make_graph(g->max_id);

    insert_vertex(path, v2);

    int child  = internal2;
    int vertex = parents[child];

    while(vertex != -1) {

        insert_vertex(path, g->hash[vertex]);

        insert_edge(path, g->hash[vertex], g->hash[child], distances[child] - distances[vertex]);
        insert_edge(path, g->hash[child], g->hash[vertex], distances[child] - distances[vertex]);

        child = vertex;
        vertex = parents[vertex];

    }

    pair* result = (pair*) malloc(sizeof(pair));

    result->first = (void*) path;
    result->second = (void*) distances[internal2];

    free(distances);
    free(parents);

    return result;

}

int max(int x, int y) {

    return (x > y) ? x : y;

}

// O(max_id + V + E)
graph* graph_union(graph* g1, graph* g2) {

    graph* g = make_graph( max(g1->max_id, g2->max_id) );

    for(int i=0; i<g1->nVertices; i++) {
        insert_vertex(g, g1->hash[i]);
    }

    for(int i=0; i<g1->nVertices; i++) {

        llist* e = g1->lst[i];
        
        while(e != NULL) {

            insert_edge(g, g1->hash[i], g1->hash[e->data], e->weight);
            e = e->next;

        }

    }

    for(int j=0; j<g2->nVertices; j++) {
        insert_vertex(g, g2->hash[j]);
    }

    for(int i=0; i<g2->nVertices; i++) {

        llist* e = g2->lst[i];
        
        while(e != NULL) {

            insert_edge(g, g2->hash[i], g2->hash[e->data], e->weight);
            e = e->next;
            
        }

    }

    return g;

}

void to_graphviz(graph* g, char* filename) {

    FILE* fp = fopen(filename, "w");

    fprintf(fp, "digraph G {\n\n");

    for(int i=0; i<g->nVertices; i++) {

        fprintf(fp, "\t%d [label=\"%d\"];\n", i, g->hash[i]);

    }

    fprintf(fp, "\n");

    for(int i=0; i<g->nVertices; i++) {

        llist* curr = g->lst[i];

        while(curr != NULL) {

            fprintf(fp, "\t%d -> %d [label=\"%d\"];\n", i, curr->data, curr->weight);  
            curr = curr->next;

        }

    }

    fprintf(fp, "\n}");

    fclose(fp);

}

void destroy_graph(graph* g) {

    if(g == NULL) {
        return;
    }

    for(int i=0; i<g->nVertices; i++) {

        destroy_llist(g->lst[i]);
        g->lst[i] = NULL;

    }

    g->capacity = 0;
    g->max_id = 0;
    g->nVertices = 0;

    free(g->hash);
    g->hash = NULL;

    free(g->deg);
    g->deg = NULL;

    free(g->lst);
    g->lst = NULL;

    free(g->reverse_hash);
    g->reverse_hash = NULL;

    free(g);

}

int steiner_verification(graph* g, int v, void* input) {

    int check1 = degree(g, v) >= 3;
    int check2 = element_exists(v, (set_t*) input);

    return check1 || check2;

}

void fill_table_under_mask(graph* g, int** table, set_t* terminals, int** distances, long long mask) {

    for(int v=0; v<g->nVertices; v++) {

        set_t* X = get_subset(terminals, mask);
        
        int w = _dfs(g, v, steiner_verification, X); // O(V+E)

        if( element_exists(g->hash[w], X) ) {

            long long submask = 1ll << (set_size(terminals) - find_position(terminals, g->hash[w]) - 1);

            table[v][mask - 1] = distances[v][w] + table[w][(mask & ~submask) - 1];   

        } else {

            int min = INT_MAX;

            for(long long submask = (mask - 1) & mask; submask != 0; submask = (submask - 1) & mask) { // iterate over submasks of the mask

                long long tmp = 1ll << (set_size(terminals) - 1);

                int cost = distances[v][w] + table[w][submask - 1] + table[w][(mask & ~submask) - 1];

                if(cost < min) {

                    min = cost;

                }

            }

            table[v][mask - 1] = min;

        }

        free(X);

    }

}

void fill_table_k_combinations(graph* g, int** table, set_t* terminals, int** distances, long long mask, int position, int k) {

    if(__builtin_popcount(mask) == k) {

        fill_table_under_mask(g, table, terminals, distances, mask);
        return;

    }

    if(position >= set_size(terminals)) {
        return;
    }

    fill_table_k_combinations(g, table, terminals, distances, mask, position + 1, k);
    fill_table_k_combinations(g, table, terminals, distances, mask | (1 << position), position + 1, k);

}

int steiner_bottom_up(graph* g, set_t* terminals) {

    int**    costs =    (int**) malloc(sizeof(int) * g->nVertices);
    graph*** trees = (graph***) malloc(sizeof(graph**) * g->nVertices);

    long long num_combinations =  (long long) pow(2, set_size(terminals)) - 1;

    for(int v=0; v<g->nVertices; v++) {

        costs[v] = (int*) malloc(sizeof(int) * num_combinations);
        trees[v] = (graph**) malloc(sizeof(graph*) * num_combinations);

    }

    // All pairs shortest path

    int** distances = (int**) malloc(sizeof(int*) * g->nVertices);
    int** parents = (int**) malloc(sizeof(int*) * g->nVertices);

    for(int i=0; i<g->nVertices; i++) {

        pair* p = dijkstra(g, i); // (E + V log (V))

        distances[i] = (int*) p->first;
        parents[i] = (int*) p->second;

        free(p);

    }

    // Fill base cases

    for(int v=0; v<g->nVertices; v++) {

        long long mask = 1ll << (set_size(terminals) - 1);

        for(int pos=set_size(terminals) - 1; pos >= 0; pos--) {

            int u = g->reverse_hash[get_element(terminals, pos)];

            costs[v][mask - 1] = distances[v][u];

            mask = mask >> 1;

        }

    }

    // Start building the array in order by incrementing a mask until 2^|T| - 1

    for(int comb=2; comb<=set_size(terminals); comb++) {

        fill_table_k_combinations(g, costs, terminals, distances, 0, 0, comb);

    }

    // Extract minimum from table.

    int min = INT_MAX;

    for(int i=0; i<g->nVertices; i++) {

        if(costs[i][num_combinations - 1] < min) {
            min = costs[i][num_combinations - 1];
        }

    }

    // Print table

    printf("\n");
    for(int i=0; i<num_combinations; i++) {
        printf("+--");
    }
    printf("+\n");

    for(int i=0; i<g->nVertices; i++) {
        printf("|");
        for(int j=0; j<num_combinations; j++) {
            printf("%2d|", costs[i][j]);
        }
        printf("\n");
        for(int j=0; j<num_combinations; j++) {
            printf("+--");
        }
        printf("+\n");
    }
    printf("\n");
    
    return min;

}