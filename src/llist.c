
#include <stdlib.h>

#include "llist.h"

llist_t* llist_add(llist_t* lst, void* data) {

    llist_t* l = (llist_t*) malloc(sizeof(llist_t));

    l->data = data;
    l->next = lst;

    return l;

}

void destroy_llist(llist_t* lst, void free_data(void*)) {

    if(lst != NULL) {

        llist_t* l = lst;
        
        while(l != NULL) {

            llist_t* del = l;
            l = l->next;

            free_data(del->data);

            del->next = NULL;
            free(del);

        }

    }
    

}