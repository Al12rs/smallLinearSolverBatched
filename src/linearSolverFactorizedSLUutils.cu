//dependencies for linearSolverFactorizedSLU_batched.cpp

#include "utils.h"
#include "magma_types.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//taken from slaswp_batched.cu

#define BLK_SIZE 256


/******************************************************************************/
// serial swap that does swapping one row by one row
__global__ void slaswp_rowserial_kernel_batched( int n, float **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    float* dA = dA_array[blockIdx.z];
    magma_int_t *dipiv = ipiv_array[blockIdx.z];
    
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    k1--;
    k2--;

    if (tid < n) {
        float A1;

        for (int i1 = k1; i1 < k2; i1++) 
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}

/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing  
extern "C" void
magma_slaswp_rowserial_batched(magma_int_t n, float** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount, cudaStream_t queue)
{
    if (n == 0) return;

    int blocks = magma_ceildiv( n, BLK_SIZE );
    dim3  grid(blocks, 1, batchCount);

    slaswp_rowserial_kernel_batched
        <<< grid, max(BLK_SIZE, n), 0, queue >>>
        (n, dA_array, lda, k1, k2, ipiv_array);
}
