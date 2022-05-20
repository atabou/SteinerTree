
#include <stdlib.h>

#include "llist.h"

llist_t* llist_add(llist_t* lst, int32_t dest, float weight) {

    llist_t* l = (llist_t*) malloc(sizeof(llist_t));

    l->dest = dest;
    l->weight = weight;
    l->next = lst;

    return l;

}

void destroy_llist(llist_t* lst) {

    if(lst != NULL) {

        llist_t* l = lst;
        
        while(l != NULL) {

            llist_t* del = l;
            l = l->next;

            del->next = NULL;
            free(del);

        }

    }
    

}
