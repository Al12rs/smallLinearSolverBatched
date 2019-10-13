#include "utils.h"
#include "utilscu.cuh"
#include "magma_types.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas/zgetrf_batched_smallsq_noshfl.cu, normal z -> s, Fri Aug  2 17:10:10 2019
*/

// for every size [1:32], how many 1D configs can a warp hold?
#define NTCOL_1D_DEFAULT 32, 16, 10, 8, 6, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
const magma_int_t ntcol_1d_default[] = {NTCOL_1D_DEFAULT};

// =============================================================================
// LU
// =============================================================================
// Kepler (or older) 
const magma_int_t sgetrf_batched_ntcol_300[] = {32, 16, 8, 8, 8, 8, 8, 32, 8, 16, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4};
const magma_int_t dgetrf_batched_ntcol_300[] = {32, 16, 8, 16, 8, 4, 4, 32, 4, 4, 8, 8, 8, 8, 8, 8, 4, 4, 16, 16, 16, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
const magma_int_t cgetrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_300[] = {NTCOL_1D_DEFAULT};

// Pascal (used also for maxwell) 
const magma_int_t sgetrf_batched_ntcol_600[] = {8, 5, 3, 2, 3, 3, 3, 3, 32, 32, 32, 32, 2, 15, 16, 16, 16, 13, 12, 10, 11, 10, 10, 8, 9, 8, 2, 6, 6, 6, 1, 1};
const magma_int_t dgetrf_batched_ntcol_600[] = {8, 4, 2, 3, 2, 2, 2, 5, 3, 3, 5, 12, 3, 3, 9, 8, 6, 6, 6, 5, 6, 6, 6, 8, 8, 8, 7, 4, 4, 4, 4, 4};
const magma_int_t cgetrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_600[] = {NTCOL_1D_DEFAULT};

// Volta  
const magma_int_t sgetrf_batched_ntcol_700[] = {8, 12, 3, 4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 8, 12, 12, 12, 10, 10, 10, 10, 8, 8, 8, 6, 6, 4, 6, 4, 1};
const magma_int_t dgetrf_batched_ntcol_700[] = {7, 4, 2, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 9, 12, 8, 8, 10, 10, 10, 10, 8, 8, 8, 7, 6, 6, 6, 4, 4};
const magma_int_t cgetrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};
const magma_int_t zgetrf_batched_ntcol_700[] = {NTCOL_1D_DEFAULT};

/// @see magma_get_zgetrf_batched_ntcol
magma_int_t magma_get_sgetrf_batched_ntcol(magma_int_t m, magma_int_t n)
{
    magma_int_t* ntcol_array; 

    if(m != n || m < 0 || m > 32) return 1;
    
    magma_int_t arch = magma_getdevice_arch();
    if      (arch <= 300) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_300; 
    else if (arch <= 600) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_600;
    else if (arch <= 700) ntcol_array = (magma_int_t*)sgetrf_batched_ntcol_700;
    else                  ntcol_array = (magma_int_t*)ntcol_1d_default; 
    
    return ntcol_array[m-1];
}

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
extern __shared__ float zdata[];
template<int N, int NPOW2>
__global__ void
sgetrf_batched_smallsq_noshfl_kernel( float** dA_array, int ldda, 
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;
    
    float* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];
    
    float rA[N] = {MAGMA_S_ZERO};
    float reg = MAGMA_S_ZERO; 
    
    int max_id, rowid = tx;
    int linfo = 0;
    float rx_abs_max = MAGMA_D_ZERO;
    
    float *sx = (float*)(zdata);
    float* dsx = (float*)(sx + blockDim.y * NPOW2);
    int* sipiv = (int*)(dsx + blockDim.y * NPOW2);
    sx    += ty * NPOW2;
    dsx   += ty * NPOW2;
    sipiv += ty * NPOW2;

    // read 
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }
        
    #pragma unroll
    for(int i = 0; i < N; i++){
        // isamax and find pivot
        dsx[ rowid ] = fabs(MAGMA_S_REAL( rA[i] )) + fabs(MAGMA_S_IMAG( rA[i] ));
        magmablas_syncwarp();
        rx_abs_max = dsx[i];
        max_id = i; 
        #pragma unroll
        for(int j = i+1; j < N; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        
        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = i; j < N; j++){
                sx[j] = rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id; 
        }
        magmablas_syncwarp();
        
        reg = MAGMA_S_DIV(MAGMA_S_ONE, sx[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        magmablas_syncwarp();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }
    // write
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);    // fortran indexing
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    sgetrf_batched_smallsq_noshfl computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges. 
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_sgetrf_batched_smallsq_noshfl( 
    magma_int_t n, 
    float** dA_array, magma_int_t ldda, 
    magma_int_t** ipiv_array, magma_int_t* info_array, 
    magma_int_t batchCount, cudaStream_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m = n;
    
    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }
    
    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    
    if( m == 0) return 0;
    
    const magma_int_t ntcol = magma_get_sgetrf_batched_ntcol(m, n);
    magma_int_t shmem  = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(float);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(float);
    dim3 threads(magma_ceilpow2(m), ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    switch(m){
        case  1: sgetrf_batched_smallsq_noshfl_kernel< 1, magma_ceilpow2( 1)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  2: sgetrf_batched_smallsq_noshfl_kernel< 2, magma_ceilpow2( 2)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  3: sgetrf_batched_smallsq_noshfl_kernel< 3, magma_ceilpow2( 3)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  4: sgetrf_batched_smallsq_noshfl_kernel< 4, magma_ceilpow2( 4)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  5: sgetrf_batched_smallsq_noshfl_kernel< 5, magma_ceilpow2( 5)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  6: sgetrf_batched_smallsq_noshfl_kernel< 6, magma_ceilpow2( 6)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  7: sgetrf_batched_smallsq_noshfl_kernel< 7, magma_ceilpow2( 7)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  8: sgetrf_batched_smallsq_noshfl_kernel< 8, magma_ceilpow2( 8)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  9: sgetrf_batched_smallsq_noshfl_kernel< 9, magma_ceilpow2( 9)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 10: sgetrf_batched_smallsq_noshfl_kernel<10, magma_ceilpow2(10)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 11: sgetrf_batched_smallsq_noshfl_kernel<11, magma_ceilpow2(11)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 12: sgetrf_batched_smallsq_noshfl_kernel<12, magma_ceilpow2(12)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 13: sgetrf_batched_smallsq_noshfl_kernel<13, magma_ceilpow2(13)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 14: sgetrf_batched_smallsq_noshfl_kernel<14, magma_ceilpow2(14)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 15: sgetrf_batched_smallsq_noshfl_kernel<15, magma_ceilpow2(15)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 16: sgetrf_batched_smallsq_noshfl_kernel<16, magma_ceilpow2(16)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 17: sgetrf_batched_smallsq_noshfl_kernel<17, magma_ceilpow2(17)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 18: sgetrf_batched_smallsq_noshfl_kernel<18, magma_ceilpow2(18)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 19: sgetrf_batched_smallsq_noshfl_kernel<19, magma_ceilpow2(19)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 20: sgetrf_batched_smallsq_noshfl_kernel<20, magma_ceilpow2(20)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 21: sgetrf_batched_smallsq_noshfl_kernel<21, magma_ceilpow2(21)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 22: sgetrf_batched_smallsq_noshfl_kernel<22, magma_ceilpow2(22)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 23: sgetrf_batched_smallsq_noshfl_kernel<23, magma_ceilpow2(23)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 24: sgetrf_batched_smallsq_noshfl_kernel<24, magma_ceilpow2(24)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 25: sgetrf_batched_smallsq_noshfl_kernel<25, magma_ceilpow2(25)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 26: sgetrf_batched_smallsq_noshfl_kernel<26, magma_ceilpow2(26)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 27: sgetrf_batched_smallsq_noshfl_kernel<27, magma_ceilpow2(27)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 28: sgetrf_batched_smallsq_noshfl_kernel<28, magma_ceilpow2(28)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 29: sgetrf_batched_smallsq_noshfl_kernel<29, magma_ceilpow2(29)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 30: sgetrf_batched_smallsq_noshfl_kernel<30, magma_ceilpow2(30)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 31: sgetrf_batched_smallsq_noshfl_kernel<31, magma_ceilpow2(31)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 32: sgetrf_batched_smallsq_noshfl_kernel<32, magma_ceilpow2(32)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}


/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas/zgetrf_batched_smallsq_shfl.cu, normal z -> s, Fri Aug  2 17:10:10 2019
*/

// This kernel uses registers for matrix storage, shared mem. and shuffle for communication.
// It also uses lazy swap.
extern __shared__ float ddata[];
template<int N, int NSHFL>
__global__ void
sgetrf_batched_smallsq_shfl_kernel( float** dA_array, int ldda, 
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;
    
    float* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];
    
    float rA[N] = {MAGMA_S_ZERO};
    float  y[N] = {MAGMA_S_ZERO};
    float reg = MAGMA_S_ZERO; 
    int max_id, current_piv_tx, rowid = tx, linfo = 0;
    float rx_abs_max = MAGMA_D_ZERO;
    // shared memory pointers
    float* sx = (float*)(ddata);
    int* sipiv = (int*)(sx + blockDim.y * NSHFL);
    sx += ty * NSHFL;
    sipiv += ty * (NSHFL+1);
    volatile int* scurrent_piv_tx = (volatile int*)(sipiv + NSHFL);    
    
    // read 
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }
        
    #pragma unroll
    for(int i = 0; i < N; i++){
        sx[ rowid ] = fabs(MAGMA_S_REAL( rA[i] )) + fabs(MAGMA_S_IMAG( rA[i] ));
        rx_abs_max = sx[i];
        max_id = i; 
        #pragma unroll
        for(int j = i; j < N; j++){
            if( sx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = sx[j];
            }
        }
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        //linfo = ( rx_abs_max == MAGMA_D_ZERO ) ? min(linfo, i+1) : 0;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            (*scurrent_piv_tx) = tx; 
        }
        else if(rowid == i){
            rowid = max_id; 
        }
        current_piv_tx = (*scurrent_piv_tx);
        
        #pragma unroll
        for(int j = i; j < N; j++){
            y[j] = magmablas_sshfl( rA[j], current_piv_tx, NSHFL);
        }
        reg = MAGMA_S_DIV(MAGMA_S_ONE, y[i] ); 
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * y[j];
            }
        }
    }
    
    // write
    if( tx == 0 ){
        (*info) = (magma_int_t)linfo;
    }
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    sgetrf_batched_smallsq_noshfl computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges. 
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_sgetrf_batched_smallsq_shfl( 
    magma_int_t n, 
    float** dA_array, magma_int_t ldda, 
    magma_int_t** ipiv_array, magma_int_t* info_array, 
    magma_int_t batchCount, cudaStream_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m = n;
    
    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }
    
    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    
    if( m == 0) return 0;
    
    const magma_int_t ntcol = magma_get_sgetrf_batched_ntcol(m, n);
    magma_int_t shmem  = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(float);
                shmem += ntcol * 1 * sizeof(int);
    dim3 threads(magma_ceilpow2(m), ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    switch(m){
        case  1: sgetrf_batched_smallsq_shfl_kernel< 1, magma_ceilpow2( 1)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  2: sgetrf_batched_smallsq_shfl_kernel< 2, magma_ceilpow2( 2)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  3: sgetrf_batched_smallsq_shfl_kernel< 3, magma_ceilpow2( 3)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  4: sgetrf_batched_smallsq_shfl_kernel< 4, magma_ceilpow2( 4)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  5: sgetrf_batched_smallsq_shfl_kernel< 5, magma_ceilpow2( 5)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  6: sgetrf_batched_smallsq_shfl_kernel< 6, magma_ceilpow2( 6)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  7: sgetrf_batched_smallsq_shfl_kernel< 7, magma_ceilpow2( 7)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  8: sgetrf_batched_smallsq_shfl_kernel< 8, magma_ceilpow2( 8)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  9: sgetrf_batched_smallsq_shfl_kernel< 9, magma_ceilpow2( 9)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 10: sgetrf_batched_smallsq_shfl_kernel<10, magma_ceilpow2(10)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 11: sgetrf_batched_smallsq_shfl_kernel<11, magma_ceilpow2(11)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 12: sgetrf_batched_smallsq_shfl_kernel<12, magma_ceilpow2(12)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 13: sgetrf_batched_smallsq_shfl_kernel<13, magma_ceilpow2(13)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 14: sgetrf_batched_smallsq_shfl_kernel<14, magma_ceilpow2(14)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 15: sgetrf_batched_smallsq_shfl_kernel<15, magma_ceilpow2(15)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 16: sgetrf_batched_smallsq_shfl_kernel<16, magma_ceilpow2(16)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 17: sgetrf_batched_smallsq_shfl_kernel<17, magma_ceilpow2(17)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 18: sgetrf_batched_smallsq_shfl_kernel<18, magma_ceilpow2(18)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 19: sgetrf_batched_smallsq_shfl_kernel<19, magma_ceilpow2(19)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 20: sgetrf_batched_smallsq_shfl_kernel<20, magma_ceilpow2(20)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 21: sgetrf_batched_smallsq_shfl_kernel<21, magma_ceilpow2(21)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 22: sgetrf_batched_smallsq_shfl_kernel<22, magma_ceilpow2(22)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 23: sgetrf_batched_smallsq_shfl_kernel<23, magma_ceilpow2(23)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 24: sgetrf_batched_smallsq_shfl_kernel<24, magma_ceilpow2(24)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 25: sgetrf_batched_smallsq_shfl_kernel<25, magma_ceilpow2(25)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 26: sgetrf_batched_smallsq_shfl_kernel<26, magma_ceilpow2(26)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 27: sgetrf_batched_smallsq_shfl_kernel<27, magma_ceilpow2(27)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 28: sgetrf_batched_smallsq_shfl_kernel<28, magma_ceilpow2(28)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 29: sgetrf_batched_smallsq_shfl_kernel<29, magma_ceilpow2(29)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 30: sgetrf_batched_smallsq_shfl_kernel<30, magma_ceilpow2(30)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 31: sgetrf_batched_smallsq_shfl_kernel<31, magma_ceilpow2(31)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 32: sgetrf_batched_smallsq_shfl_kernel<32, magma_ceilpow2(32)><<<grid, threads, shmem, queue >>>(dA_array, ldda, ipiv_array, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}
