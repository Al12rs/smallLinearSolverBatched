#include "utils.h"

#if defined( _WIN32 ) || defined( _WIN64 )
#  include <time.h>
#  include <sys/timeb.h>
#  include <windows.h>
#  if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#  else
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#  endif
#else
#  include <sys/time.h>
#endif


static int g_init = 0;
int g_magma_devices_cnt = 0;
struct magma_device_info
{
    size_t memory;
    size_t shmem_block;      // maximum shared memory per thread block in bytes
    size_t shmem_multiproc;  // maximum shared memory per multiprocessor in bytes
    magma_int_t cuda_arch;
    magma_int_t multiproc_count;    // number of multiprocessors
};
struct magma_device_info* g_magma_devices = NULL;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/***************************************************************************//**
    Allocate size bytes on CPU.
    The purpose of using this instead of malloc is to properly align arrays
    for vector (SSE, AVX) instructions. The default implementation uses
    posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
    to align memory to a 64 byte boundary (typical cache line size).
    Use magma_free_cpu() to free this memory.

    @param[out]
    ptrPtr  On output, set to the pointer that was allocated.
            NULL on failure.

    @param[in]
    size    Size in bytes to allocate. If size = 0, allocates some minimal size.

    @return MAGMA_SUCCESS
    @return MAGMA_ERR_HOST_ALLOC on failure

    Type-safe versions avoid the need for a (void**) cast and explicit sizeof.
    @see magma_smalloc_cpu
    @see magma_dmalloc_cpu
    @see magma_cmalloc_cpu
    @see magma_zmalloc_cpu
    @see magma_imalloc_cpu
    @see magma_index_malloc_cpu

    @ingroup magma_malloc_cpu
*******************************************************************************/
extern "C" int
magma_malloc_cpu(void** ptrPtr, size_t size)
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if (size == 0)
        size = sizeof(cuDoubleComplex);
#if 1
#if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = _aligned_malloc(size, 64);
    if (*ptrPtr == NULL) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#else
    int err = posix_memalign(ptrPtr, 64, size);
    if (err != 0) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
#else
    * ptrPtr = malloc(size);
    if (*ptrPtr == NULL) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    return MAGMA_SUCCESS;
}


// =============================================================================
// initialization

/***************************************************************************//**
    Initializes the MAGMA library.
    Caches information about available CUDA devices.

    Every magma_init call must be paired with a magma_finalize call.
    Only one thread needs to call magma_init and magma_finalize,
    but every thread may call it. If n threads call magma_init,
    the n-th call to magma_finalize will release resources.

    When renumbering CUDA devices, call cudaSetValidDevices before calling magma_init.
    When setting CUDA device flags, call cudaSetDeviceFlags before calling magma_init.

    @retval MAGMA_SUCCESS
    @retval MAGMA_ERR_UNKNOWN
    @retval MAGMA_ERR_HOST_ALLOC

    @see magma_finalize

    @ingroup magma_init
*******************************************************************************/
extern "C" int
magma_init()
{
    int info = 0;

    {
        if (g_init == 0) {
            // query number of devices
            cudaError_t err;
            g_magma_devices_cnt = 0;
            err = cudaGetDeviceCount(&g_magma_devices_cnt);
            if (err != 0 && err != cudaErrorNoDevice) {
                info = UNSPECIFIED_ERR;
                goto cleanup;
            }

            // allocate list of devices
            size_t size;
            size = max(1, g_magma_devices_cnt) * sizeof(struct magma_device_info);
            magma_malloc_cpu((void**)&g_magma_devices, size);
            if (g_magma_devices == NULL) {
                info = MAGMA_ERR_HOST_ALLOC;
                goto cleanup;
            }
            memset(g_magma_devices, 0, size);

            // query each device
            for (int dev = 0; dev < g_magma_devices_cnt; ++dev) {
                cudaDeviceProp prop;
                err = cudaGetDeviceProperties(&prop, dev);
                if (err != 0) {
                    info = MAGMA_ERR_UNKNOWN;
                }
                else {
                    g_magma_devices[dev].memory = prop.totalGlobalMem;
                    g_magma_devices[dev].cuda_arch = prop.major * 100 + prop.minor * 10;
                    g_magma_devices[dev].shmem_block = prop.sharedMemPerBlock;
                    g_magma_devices[dev].shmem_multiproc = prop.sharedMemPerMultiprocessor;
                    g_magma_devices[dev].multiproc_count = prop.multiProcessorCount;
                }
            }

#ifdef HAVE_PTHREAD_KEY
            // create thread-specific key
            // currently, this is needed only for MAGMA v1 compatability
            // see magma_init, magmablas(Set|Get)KernelStream, magmaGetQueue
            info = pthread_key_create(&g_magma_queue_key, NULL);
            if (info != 0) {
                info = MAGMA_ERR_UNKNOWN;
                goto cleanup;
            }
#endif
        }
    cleanup:
        g_init += 1;  // increment (init - finalize) count
    }

    return info;
}




// error handler called in case of invalid value parameters
void utils_reportError(const char* srname, int neg_info)
{
    // the first 3 cases are unusual for calling xerbla;
    // normally runtime errors are passed back in info.
    if (neg_info < 0) {
        fprintf(stderr, "Error in %s, function-specific error (info = %lld)\n",
            srname, (long long)-neg_info);
    }
    else if (neg_info == 0) {
        fprintf(stderr, "No error, why is %s calling xerbla? (info = %lld)\n",
            srname, (long long)-neg_info);
    }
    else if (neg_info >= UNSPECIFIED_ERR) {
        fprintf(stderr, "Error in %s, %s (info = %lld)\n",
            srname, "unknown error", (long long)-neg_info);
    }
    else {
        // this is the normal case for calling reportError;
        // invalid parameter values are usually logic errors, not runtime errors.
        fprintf(stderr, "On entry to %s, parameter %lld had an illegal value (info = %lld)\n",
            srname, (long long)neg_info, (long long)-neg_info);
    }
}

/***************************************************************************//**
    Prints error message to stderr.
    C++ function overloaded for different error types (CUDA,
    cuBLAS, MAGMA errors). Note CUDA and cuBLAS errors are enums,
    so can be differentiated.
    Used by the check_error() and check_xerror() macros.

    @param[in]
    err     Error code.

    @param[in]
    func    Function where error occurred; inserted by check_error().

    @param[in]
    file    File     where error occurred; inserted by check_error().

    @param[in]
    line    Line     where error occurred; inserted by check_error().

    @ingroup magma_error_internal
*******************************************************************************/
void utils_xerror(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
            cudaGetErrorString(err), err, func, file, line);
    }
}

/***************************************************************************//**
    Allocates memory on the GPU. CUDA imposes a synchronization.
    Use magma_free() to free this memory.

    @param[out]
    ptrPtr  On output, set to the pointer that was allocated.
            NULL on failure.

    @param[in]
    size    Size in bytes to allocate. If size = 0, allocates some minimal size.

    @return RET_SUCCESS
    @return RET_ERR_DEVICE_ALLOC on failure

    Type-safe versions avoid the need for a (void**) cast and explicit sizeof.
    @see magma_smalloc
    @see magma_dmalloc
    @see magma_cmalloc
    @see magma_zmalloc
    @see magma_imalloc
    @see magma_index_malloc

    @ingroup magma_malloc
*******************************************************************************/
extern "C" int
magma_malloc(void** ptrPtr, size_t size)
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if (size == 0)
        size = sizeof(cuDoubleComplex);
    if (cudaSuccess != cudaMalloc(ptrPtr, size)) {
        return RES_ERR_DEVICE_ALLOC;
    }
    return RES_SUCCESS;
}


/***************************************************************************//**
    Returns CUDA architecture capability for the current device.
    This requires magma_init() to be called first to cache the information.
    Version is an integer xyz, where x is major, y is minor, and z is micro,
    the same as __CUDA_ARCH__. Thus for architecture 1.3.0 it returns 130.

    @return CUDA_ARCH for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" int
utils_getdevice_arch()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice(&dev);
    check_error(err);
    ((void)(err));
    if (g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt) {
        fprintf(stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__);
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}


extern "C" int
magma_getdevice_arch()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice(&dev);
    check_error(err);
    ((void)(err));
    if (g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt) {
        fprintf(stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__);
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}


void magma_xerror(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
            cudaGetErrorString(err), err, func, file, line);
    }
}

extern "C" void
magma_queue_sync_internal(
    cudaStream_t queue,
    const char* func, const char* file, int line)
{
    cudaError_t err;
    err = cudaStreamSynchronize(queue);
    check_xerror(err, func, file, line);
    MAGMA_UNUSED(err);
}

extern "C" magma_int_t
magma_free_internal(magma_ptr ptr,
    const char* func, const char* file, int line)
{
#ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if (ptr != NULL && g_pointers_dev.count(ptr) == 0) {
        fprintf(stderr, "magma_free( %p ) that wasn't allocated with magma_malloc.\n", ptr);
    }
    else {
        g_pointers_dev.erase(ptr);
    }
    g_pointers_mutex.unlock();
#endif

    cudaError_t err = cudaFree(ptr);
    check_xerror(err, func, file, line);
    if (err != cudaSuccess) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}


extern "C"
void magma_xerbla(const char* srname, magma_int_t neg_info)
{
    // the first 3 cases are unusual for calling xerbla;
    // normally runtime errors are passed back in info.
    if (neg_info < 0) {
        fprintf(stderr, "Error in %s, function-specific error (info = %lld)\n",
            srname, (long long)-neg_info);
    }
    else if (neg_info == 0) {
        fprintf(stderr, "No error, why is %s calling xerbla? (info = %lld)\n",
            srname, (long long)-neg_info);
    }
    else {
        // this is the normal case for calling xerbla;
        // invalid parameter values are usually logic errors, not runtime errors.
        fprintf(stderr, "On entry to %s, parameter %lld had an illegal value (info = %lld)\n",
            srname, (long long)neg_info, (long long)-neg_info);
    }
}

extern "C" magma_int_t
magma_free_cpu(void* ptr)
{
#ifdef DEBUG_MEMORY
    g_pointers_mutex.lock();
    if (ptr != NULL && g_pointers_cpu.count(ptr) == 0) {
        fprintf(stderr, "magma_free_cpu( %p ) that wasn't allocated with magma_malloc_cpu.\n", ptr);
    }
    else {
        g_pointers_cpu.erase(ptr);
    }
    g_pointers_mutex.unlock();
#endif

#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    return MAGMA_SUCCESS;
}


extern "C" magma_int_t
magma_finalize()
{
    magma_int_t info = 0;

    //g_mutex.lock();
    {
        if (g_init <= 0) {
            info = MAGMA_ERR_NOT_INITIALIZED;
        }
        else {
            g_init -= 1;  // decrement (init - finalize) count
            if (g_init == 0) {
                info = 0;

                if (g_magma_devices != NULL) {
                    magma_free_cpu(g_magma_devices);
                    g_magma_devices = NULL;
                }

#ifndef MAGMA_NO_V1
                

#ifdef HAVE_PTHREAD_KEY
                pthread_key_delete(g_magma_queue_key);
#endif
#endif // MAGMA_NO_V1

#ifdef DEBUG_MEMORY
                magma_warn_leaks(g_pointers_dev, "device");
                magma_warn_leaks(g_pointers_cpu, "CPU");
                magma_warn_leaks(g_pointers_pin, "CPU pinned");
#endif
            }
        }
    }
    //g_mutex.unlock();

    return info;
}



// =============================================================================
// Emulate gettimeofday on Windows.

#if defined( _WIN32 ) || defined( _WIN64 )
#ifndef _TIMEZONE_DEFINED
#define _TIMEZONE_DEFINED
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};
#endif

extern "C"
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME         ft;
    unsigned __int64 tmpres = 0;
    static int       tzflag = 0;

    if (NULL != tv) {
        GetSystemTimeAsFileTime(&ft);
        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres /= 10;  /*convert into microseconds*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;

        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }
    if (NULL != tz) {
        if (!tzflag) {
            _tzset();
            tzflag = 1;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }
    return 0;
}
#endif


extern "C"
double magma_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

#undef min
#undef max
