#ifndef UTILS_H
#define UTILS_H

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "magma_types.h"


#define UNSPECIFIED_ERR      -100     ///< unspecified error
#define MAGMA_ERR_UNKNOWN    -116     ///< unspecified error
#define RES_SUCCESS           0
#define MAGMA_SUCCESS         0
#define RES_ERR_DEVICE_ALLOC -113
#define UTILS_ERR_UNKNOWN    -116
#define MAGMA_ERR_HOST_ALLOC -112     ///< could not malloc CPU host memory

/*
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
*/
struct magma_device_info;

#if __cplusplus
extern "C" {
#endif

    int magma_malloc_cpu(void** ptrPtr, size_t size);
    int magma_init();
    int magma_malloc(void** ptrPtr, size_t size);
    /// Type-safe version of magma_malloc(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
    static inline int magma_imalloc(magmaInt_ptr* ptr_ptr, size_t n) { return magma_malloc((magma_ptr*)ptr_ptr, n * sizeof(magma_int_t)); }
    int utils_getdevice_arch();
    int magma_getdevice_arch();
    void magma_queue_sync_internal(
        cudaStream_t queue,
        const char* func, const char* file, int line);
    magma_int_t magma_free_internal(magma_ptr ptr,
            const char* func, const char* file, int line);
    void magma_xerbla(const char* srname, magma_int_t neg_info);
    magma_int_t magma_free_cpu(void* ptr);
    magma_int_t magma_finalize();
    /// Type-safe version of magma_malloc_cpu(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
    static inline magma_int_t magma_imalloc_cpu(magma_int_t** ptr_ptr, size_t n) { return magma_malloc_cpu((void**)ptr_ptr, n * sizeof(magma_int_t)); }

    /// Type-safe version of magma_malloc_cpu(), for float arrays. Allocates n*sizeof(float) bytes.
    static inline magma_int_t magma_smalloc_cpu(float** ptr_ptr, size_t n) { return magma_malloc_cpu((void**)ptr_ptr, n * sizeof(float)); }

    double magma_wtime(void);
#if __cplusplus
}
#endif

void utils_reportError(const char* srname, int neg_info);
void utils_xerror(cudaError_t err, const char* func, const char* file, int line);

void magma_xerror(cudaError_t err, const char* func, const char* file, int line);

#define check_error( err ) \
        utils_xerror( err, __func__, __FILE__, __LINE__ )







#define check_xerror( err, func, file, line ) \
        magma_xerror( err, func, file, line )

#define MAGMA_UNUSED(var)  ((void)var)




#define magma_queue_sync( queue ) \
        magma_queue_sync_internal( queue, __func__, __FILE__, __LINE__ )



#define magma_free( ptr ) \
        magma_free_internal( ptr, __func__, __FILE__, __LINE__ )






/// Type-safe version of magma_malloc(), for float arrays. Allocates n*sizeof(float) bytes.
static inline magma_int_t magma_smalloc(magmaFloat_ptr* ptr_ptr, size_t n) 
{ 
    return magma_malloc((magma_ptr*)ptr_ptr, n * sizeof(float)); 
}

/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
/// @ingroup magma_ceildiv
__host__ __device__
static inline magma_int_t magma_ceildiv(magma_int_t x, magma_int_t y)
{
    return (x + y - 1) / y;
}

__host__ __device__
static inline magma_int_t magma_roundup(magma_int_t x, magma_int_t y)
{
    return magma_ceildiv(x, y) * y;
}

#define magma_ceilpow2(N)    ( (N >  16)? 32 : \
                               (N >   8)? 16 : \
                               (N >   4)?  8 : \
                               (N >   2)?  4 : \
                               (N >   0)?  2 : 0 )   


#endif //UTILS_H 
