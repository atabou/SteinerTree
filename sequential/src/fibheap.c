
#include <stdbool.h>
#include <stdlib.h>

#include "heap.h"
#include "fibheap.h"

typedef struct fibnode fibnode;
typedef struct fibheap fibheap;

void    fibheap_insert(heap* h, int n, int key);
int fibheap_empty(heap* h);
int     fibheap_find_min(heap* h);
int     fibheap_extract_min(heap* h);
heap*   fibheap_union(heap* h1, heap* h2);
void    fibheap_decrease_key(heap* h, int n, int key);
void    fibheap_delete(heap* h, int n);
void    destroy_fibheap(heap* h);

struct fibnode {

    int         id; // id of the node
    int         key; // key of the node
    bool        marked; // Specifies if the node is marked.

    fibnode*    prev; // next node in the level
    fibnode*    next; // previous node in the level

    fibnode*    parent; // parent of this fibnode

    int         rank; // number of children of the node
    fibnode*    children; // set of childrens
    
};

struct fibheap {

    fibnode*    root; // root of the fib heap (min element).
    int         rank; // max rank of any node in the fib heap.
    
    fibnode**   hash; // hash table linking the ids to the respective fibnode.
    int         hsize; // size of the hash table.

};

heap* make_fibheap(int capacity) {

    heap* h = (heap*) malloc(sizeof(heap));

    fibheap* fib = (fibheap*) malloc(sizeof(fibheap));

    fib->root   = NULL;
    fib->rank   = 0;

    fib->hash   = (fibnode**) malloc(sizeof(fibnode*) * capacity);
    fib->hsize  = capacity;

    for(int i=0; i<fib->hsize; i++) {
        fib->hash[i] = NULL;
    }

    h->data = h;

    h->insert = fibheap_insert;
    h->empty = fibheap_empty;
    h->find_min = fibheap_find_min;
    h->extract_min = fibheap_extract_min;
    h->heap_union = fibheap_union;
    h->decrease_key = fibheap_decrease_key;
    h->delete_node = fibheap_delete;
    h->destroy = destroy_fibheap;

    return h;

}

void detach_heap(fibnode* n) {

    n->next->prev = n->prev;
    n->prev->next = n->next;

    n->next = NULL;
    n->prev = NULL;

}

void attach_strand(fibnode* base, fibnode* start, fibnode* end) {

    start->prev = base;
    end->next = base->next;

    base->next->prev = end;
    base->next = start;

}

void fibheap_insert(heap* h, int id, int key) {

    fibheap* fib = (fibheap*) h->data;

    if(fib->hash[id] == NULL) {

        fibnode* fn = (fibnode*) malloc(sizeof(fibnode));

        fn->id          = id;
        fn->key         = key;
        fn->parent      = NULL;
        fn->children    = NULL;
        fn->rank        = 0; 

        if(fib->root = NULL) {

            fib->root = fn;

            fib->root->next = fib->root;
            fib->root->prev = fib->root;

        } else {

            attach_strand(fib->root, fn, fn);

            if(key < fib->root->key) {

                fib->root = fn;

            }        

        }

        fib->hash[id] = fn;

    }

}

int fibheap_empty(heap* h) {

    return (((fibheap*) h->data)->root == NULL) ? 1 : 0;

}

int fibheap_find_min(heap* h) {
    
    fibheap* fib = (fibheap*) h->data;
    return (fib->root == NULL) ? -1 : fib->root->id;

}

void detach_root(fibheap* fib) {

    if(fib->root == fib->root->next) {

        fib->root->next = NULL;
        fib->root->prev = NULL;

        fib->root = NULL;
        fib->rank = 0;

    } else {

        fibnode* ptr = fib->root->next;

        detach_heap(fib->root);

        fib->root = ptr;

    }
    

}

void update_minimum_and_rank(fibheap* fib) {

    if(fib->root != NULL) {

        fibnode* curr = fib->root->next;
        
        while(curr != fib->root) {

            if(curr->key < fib->root->key) {
                fib->root = curr;
            }

            if(curr->rank > fib->rank) {
                fib->rank = curr->rank;
            }

            curr = curr->next;

        }

    }

}

void upgrade_children(fibnode* target) {

    if(target != NULL && target->children != NULL) {

        target->children->next->prev = target->next;
        target->next = target->children->next;

        target->children->next = target;
        target->next = target->children;

        target->children = NULL;

    }

}

void downgrade_heap(fibheap* fib, fibnode** degrees, fibnode* parent, fibnode* child) {

    // Detach child

    detach_heap(child);

    // Attach child to parent childrens

    if(parent->rank == 0) {

        child->next = child;
        child->prev = child;

        parent->children = child;

    } else {

        attach_strand(parent->children, child, child);

    }

    parent->rank = parent->rank + 1;
    child->parent = parent;

    if(fib->rank < parent->rank) {

        degrees = realloc(degrees, sizeof(fibnode*) * parent->rank);
        degrees[parent->rank - 1] = parent;
        fib->rank = parent->rank;
    
    }

}

void consolidate (fibheap* fib) {

    // Initialize a hashtable.

    fibnode** degrees = (fibnode**) malloc(sizeof(fibnode*) * fib->rank);

    for(int i=0; i<fib->rank; i++) {
        degrees[i] = NULL;
    }

    // Consolidate until no heaps with the same degree exist.

    int consolidated = 0;

    while(!consolidated) {

        consolidated = 1;

        fibnode* curr = fib->root; // TODO: this will never enter the loop
        
        while(curr != fib->root) {

            if(degrees[curr->rank] == NULL) {

                degrees[curr->rank] = curr;

            } else if(curr != degrees[curr->rank]) {

                fibnode* found = degrees[curr->rank];
                degrees[curr->rank] = NULL;

                if(found->key < curr->key) {

                    downgrade_heap(fib, degrees, found, curr);

                } else {

                    downgrade_heap(fib, degrees, curr, found);

                }

                consolidated = 0;
                break;

            }

            curr = curr->next;

        }

    }

    free(degrees);

}

int fibheap_extract_min(heap* h) {

    fibheap* fib = (fibheap*) h->data;

    int minimum = -1;

    if(fib->root != NULL) {

        fibnode* extract = fib->root;

        if(extract->children != NULL) {

            upgrade_children(extract);

        }

        detach_root(fib);

        // update rank(H) to new maximum and root to new minimum.
        
        update_minimum_and_rank(fib);

        // Consolidate the heap.

        consolidate(fib);

        fib->hash[extract->id] = NULL;
        minimum = extract->id;

        extract->next = NULL;
        extract->prev = NULL;

        free(extract);

    }

    return minimum;

}

void cut(fibheap* fib, fibnode* x) {

    fibnode* p = x->parent;

    // Remove the node from the parent and children's list.

    if(x == p->children) {

        p->children = p->children->next;

        if(x == p->children) {
            
            p->children = NULL;
        
        }

    }

    detach_heap(x);
    p->rank = p->rank - 1;

    // Insert the decreased key at the top level

    attach_strand(fib->root, x, x);

    // Unmark the decreased node.

    x->marked = false;

}

void cascade(fibheap* fib, fibnode* p) {

    if(p->parent != NULL) {

        if(p->marked == false) {

            p->marked = true;
        
        } else {

            cut(fib, p);
            cascade(fib, p->parent);
        
        }

    }

}

void fibheap_decrease_key(heap* h, int n, int key) {

    fibheap* fib = (fibheap*) h->data;

    if(n < fib->hsize) {

        fibnode* decreased = fib->hash[n];
        fibnode* parent = decreased->parent;
        decreased->key = key;

        if(decreased->key < fib->root->key) {
            fib->root = decreased;
        }

        if(parent != NULL && decreased->key < parent->key) {

            cut(fib, decreased);
            cascade(fib, parent);

        }
        
    
    }

}

heap* fibheap_union(heap* h1, heap* h2) {

    return NULL;

}

void fibheap_delete(heap* h, int n) {
    return;
}

void destroy_fibheap(heap* h) {

    fibheap* fib = (fibheap*) h->data;

    if(fib->root != NULL) {

        for(int i=0; i<fib->hsize; i++) {

            if(fib->hash[i] != NULL) {

                fib->hash[i]->parent = NULL;
                fib->hash[i]->children = NULL;
                fib->hash[i]->prev = NULL;
                fib->hash[i]->next = NULL;

                free(fib->hash[i]);

            }

        }

        
    }

    fib->root = NULL;
    fib->hsize = 0;
    free(fib->hash);
    fib->hash = NULL;
    fib->hsize = 0;

    free(h->data);
    h->data = NULL;

    h->insert = NULL;
    h->find_min = NULL;
    h->extract_min = NULL;
    h->heap_union = NULL;
    h->decrease_key = NULL;
    h->delete_node = NULL;
    h->destroy = NULL;

}