
#include <stdlib.h>

#include "llist.h"

llist* llist_add(llist* lst, void* data) {

    llist* l = (llist*) malloc(sizeof(llist));

    l->data = data;
    l->next = lst;

    return l;

}

void destroy_llist(llist* lst, void free_data(void*)) {

    if(lst != NULL) {

        llist* l = lst;
        
        while(l != NULL) {

            llist* del = l;
            l = l->next;

            free_data(del->data);

            del->next = NULL;
            free(del);

        }

    }
    

}