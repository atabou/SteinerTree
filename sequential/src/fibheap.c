
#include <stdlib.h>

#include "fibheap.h"

typedef struct fibnode fibnode;

struct fibnode {

    int         id; // id of the node
    int         key; // key of the node
    
    fibnode*    prev; // next node in the level
    fibnode*    next; // previous node in the level

    int         rank; // number of children of the node
    fibnode**   children; // set of childrens
    
};

struct fibheap {

    fibnode*    root; // root of the fib heap (min element) 
    int         rank; // max rank of any node in the fib heap.
    fibnode**   marked; // set of marked nodes
    int         marks; // number of marked nodes

};

fibheap* make_fibheap() {

    fibheap* fib = (fibheap*) malloc(sizeof(fibheap));

    fib->root   = NULL;
    fib->rank   = 0;
    fib->marked = NULL;
    fib->marks  = 0;

    return fib;

}

void fibheap_insert(fibheap* fib, int n, int key) {

    fibnode* fn = (fibnode*) malloc(sizeof(fibnode));

    fn->id          = n;
    fn->key         = key;
    fn->children    = NULL;
    fn->rank        = 0; 

    if(fib->root = NULL) {

        fib->root = fn;

        fib->root->next = fib->root;
        fib->root->prev = fib->root;

    } else {

        fn->next = fib->root->next;
        fn->prev = fib->root;

        fn->next->prev = fn;
        fn->prev->next = fn;  

        if(fib->root->key > key) {

            fib->root = fn;

        }        

    }

}

int fibheap_find_min(fibheap* fib) {

    return fib->root->id;

}

int fibheap_extract_min(fibheap* fib) {

    int minimum = -1;

    if(fib->root != NULL) {

        fibnode* extract = fib->root;

        if(extract->children != NULL) {

            extract->children[extract->rank - 1]->next = extract->next;
            extract->next->prev = extract->children[extract->rank - 1];

            extract->next = extract->children[0];
            extract->children[0]->prev = extract;

            for(int i=0; i<extract->rank; i++) {
                extract->children[i] = NULL;
            }

            free(extract->children);

        }

        extract->next->prev = extract->prev;
        extract->prev->next = extract->next;

        if(extract->next == extract) {

            fib->root = NULL;
            fib->rank = 0;
            fib->marked = NULL;
            fib->marks = 0;

        } else {

            fib->root = extract->next;
            fib->rank = fib->root->rank;

            // update rank(H) to new maximum and root to new minimum.
            fibnode* curr = fib->root;
            while(curr != extract->next) {

                if(curr->key < fib->root->key) {
                    fib->root = curr;
                }

                if(curr->rank > fib->rank) {
                    fib->rank = curr->rank;
                }

                curr = curr->next;

            }

            // Consolidate.

            fibnode** degrees = (fibnode**) malloc(sizeof(fibnode*) * fib->rank);

            for(int i=0; i<fib->rank; i++) {
                degrees[i] = NULL;
            }

            int consolidated = 0;

            while(!consolidated) {

                consolidated = 1;

                fibnode* curr = fib->root; // TODO this will never enter the loop
                
                while(curr != fib->root) {

                    if(degrees[curr->rank] == NULL) {

                        degrees[curr->rank] = curr;

                    } else {

                        int k = curr->rank;
                        fibnode* found = degrees[k];

                        if(found->key > curr->key) {

                            curr->next->prev = curr->prev;
                            curr->prev->next = curr->next;

                            if(found->children == NULL) {
                                
                                found->children = (fibnode**) malloc(sizeof(fibnode*));

                                curr->next = curr;
                                curr->prev = curr;
                            
                            } else {
                            
                                realloc(found->children, sizeof(fibnode**) * (found->rank + 1));

                                curr->next = found->children[0];
                                curr->prev = found->children[found->rank - 1];

                            }

                            found->children[found->rank] = curr;
                            found->rank = found->rank + 1;

                            if(fib->rank < found->rank) {
                                realloc(degrees, found->rank * sizeof(fibnode*));
                                degrees[found->rank - 1] = NULL;
                            }

                        } else {

                            found->next->prev = found->prev;
                            found->prev->next = found->next;

                            if(curr->children == NULL) {
                                
                                curr->children = (fibnode**) malloc(sizeof(fibnode*));

                                found->next = found;
                                found->prev = found;
                            
                            } else {
                            
                                realloc(found->children, sizeof(fibnode**) * (curr->rank + 1));

                                found->next = curr->children[0];
                                found->prev = curr->children[curr->rank - 1];

                            }

                            curr->children[curr->rank] = curr;
                            curr->rank = curr->rank + 1;

                            if(fib->rank < curr->rank) {

                                realloc(degrees, curr->rank * sizeof(fibnode*));
                                degrees[curr->rank - 1] = NULL;

                            }

                        }

                        degrees[k] = NULL;
                        curr = fib->root;
                        consolidated = 0;

                        break;

                    }

                    curr = curr->next;

                }

            }

            free(degrees);



        }

        minimum = extract->id;

        extract->next = NULL;
        extract->prev = NULL;

        free(extract);

    }

    return minimum;

}

fibheap*    fibheap_union(fibheap* fib1, fibheap* fib2);

int         fibheap_decrease_key(fibheap* fib, int n, int key);

int         fibheap_delete(fibheap* fib, int n);

void        destroy_fibheap(fibheap* fib);