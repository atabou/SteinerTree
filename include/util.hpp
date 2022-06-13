/** 
 * \addtogroup Util
 * @{ */

#ifndef UTIL_H

    #define UTIL_H

    #include <time.h>

    /** Global variable that will be used by the time macro. */
    extern clock_t CLOCKMACRO;

    #define TIME(EXPR, STR) CLOCKMACRO = clock(); EXPR; printf(STR); printf(" %fs\n", (double) (clock() - CLOCKMACRO) / CLOCKS_PER_SEC)

#endif
/**@}*/
