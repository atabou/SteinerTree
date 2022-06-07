/** 
 * \addtogroup Util
 * @{ */

#ifndef UTIL_H

    #define UTIL_H

    #include <time.h>

    /** Global variable that will be used by the time macro. */
    extern clock_t CLOCKMACRO;

    #define TIME(EXPR, STR) CLOCKMACRO = clock(); EXPR; printf(STR); printf(" %fs\n", (double) (clock() - CLOCKMACRO) / CLOCKS_PER_SEC)

    /**
     * @brief Compares the value in the target pointer with the provided value, and saves the minimum between the two atomically.
     * 
     * @param target A pointer to a floating point value.
     * @param val The value containing a potential new minimum.
     * @return The old value of the pointer.
     */
    __device__ float atomicMin(float* target, float val);

#endif
/**@}*/
