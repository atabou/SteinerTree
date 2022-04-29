
#include <stdio.h>

#include "cudallist.cuh"

cudallist_t* copy_cudallist(llist_t* lst, uint32_t size) {

    if(lst == NULL) {

        return NULL;
    
    } else {
        
        cudallist_t tmp[size];

        uint32_t count;
        llist_t* curr;

        while(curr != NULL) {

            tmp[count].dest = curr->dest;
            tmp[count].weight = curr->weight;

            curr = curr->next;
            count++;

        }

        cudaError_t  err;
        cudallist_t* prev = NULL;

        for(uint32_t i=size - 1; i >= 0; i--) {

            tmp[i].next = prev;

            err = cudaMalloc(&prev, sizeof(cudallist_t));
            
            if(err) {
                printf("Could not allocate memory for cuda llist. (Error code: %d)\n", err);
                exit(err);
            }

            cudaMemcpy(prev, &(tmp[i]), sizeof(cudallist_t), cudaMemcpyHostToDevice);

        }

        return prev;
    
    }

}

void free_cudallist(cudallist_t* lst) {

    if(lst != NULL) {

        cudallist_t* curr = lst;
        cudallist_t tmp;

        while(curr != NULL) {

            cudaMemcpy(&tmp, curr, sizeof(cudallist_t), cudaMemcpyDeviceToHost);  

            cudaError_t err;

            cudaFree(curr);

            if(err) {
                printf("Could not delete cuda llist. (Error code: %d)", err);
                exit(err);
            }

            curr = tmp.next;

        }

    }


}


