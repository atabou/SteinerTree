
#include <stdio.h>

extern "C" {
    #include "llist.cuda.h"
}

cudallist_t* copy_cudallist(llist_t* lst, int32_t size) {

    if(lst == NULL) {

        return NULL;
    
    } else {
        
        cudallist_t tmp[size];

        int32_t count = 0;
        llist_t* curr = lst;

        while(curr != NULL) {

            tmp[count].dest = curr->dest;
            tmp[count].weight = curr->weight;

            curr = curr->next;
            count++;

        }

        cudaError_t  err;
        cudallist_t* prev = NULL;

        for(int32_t i=size - 1; i >= 0; i--) {

            tmp[i].next = prev;

            err = cudaMalloc(&prev, sizeof(cudallist_t));
 
            if(err) {
                printf("Could not allocate memory for cuda llist. (Error code: %d)\n", err);
                exit(err);
            }

            err = cudaDeviceSynchronize();

            if(err) {
                printf("Could not synchronize cuda device after llist allocation. (Error code: %d)\n", err);
                exit(err);
            }

            cudaMemcpy(prev, &(tmp[i]), sizeof(cudallist_t), cudaMemcpyHostToDevice);

            err = cudaDeviceSynchronize();

            if(err) {
                printf("Could not synchronize cuda device after llist memory copy. (Error code: %d)\n", err);
                exit(err);
            }

        }

        return prev;
    
    }

}

void free_cudallist(cudallist_t* lst) {

    if(lst != NULL) {

        cudaError_t err;
        cudallist_t* curr = lst;
        cudallist_t tmp;

        while(curr != NULL) {

            cudaMemcpy(&tmp, curr, sizeof(cudallist_t), cudaMemcpyDeviceToHost);  

            err = cudaDeviceSynchronize();

            if(err) {
                printf("Could not synchronize after llist memory copy in free llist. (Cuda error: %d)\n", err);
                exit(err);
            }

            err = cudaFree(curr);

            if(err) {
                printf("Could not delete cuda llist. (Error code: %d)", err);
                exit(err);
            }

            err = cudaDeviceSynchronize();

            if(err) {
                printf("Could not synchronize after llist memory free in free llist. (Cuda error: %d)\n", err);
                exit(err);
            }


            curr = tmp.next;

        }

    }


}


