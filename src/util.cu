#include "util.h"

__device__ float atomicMin(float* target, float val) {

    int32_t ret = __float_as_int(*target);

    while(val < __int_as_float(ret)) {

        int32_t old = ret;
        ret = atomicCAS((int32_t*) target, old, __float_as_int(val));
        if(ret == old){
            break;
        }

    }

    return __int_as_float(ret);

}