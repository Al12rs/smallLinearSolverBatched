#ifndef UTILSCU_CUH
#define UTILSCU_CUH

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

__device__ static inline void magmablas_syncwarp()
{
#if __CUDACC_VER_MAJOR__ >= 9
    __syncwarp();
#else
    // assume implicit warp synchronization
    // using syncthreads() is not safe here
    // as the warp can be part of a bigger thread block
#endif
}

#define SHFL_FULL_MASK 0xffffffff

__device__ static inline float magmablas_sshfl(float var, int srcLane, int width = 32, unsigned mask = SHFL_FULL_MASK)
{
#if __CUDA_ARCH__ >= 300
#if __CUDACC_VER_MAJOR__ < 9
    return __shfl(var, srcLane, width);
#else
    return __shfl_sync(mask, var, srcLane, width);
#endif
#else    // pre-Kepler GPUs
    return MAGMA_S_ZERO;
#endif
}


#endif //UTILSCU_CUH
