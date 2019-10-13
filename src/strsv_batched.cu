

#include "utils.h"
#include "magma_types.h"
#include "operation_batched.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


/*
    -- MAGMA (version 2.5.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2019

       @author Tingxing Dong
       @author Azzam Haidar

       @generated from magmablas/ztrsv_batched.cu, normal z -> s, Fri Aug  2 17:10:10 2019
*/


#define PRECISION_s

#define NB 256  //NB is the 1st level blocking in recursive blocking, BLOCK_SIZE is the 2ed level, NB=256, BLOCK_SIZE=64 is optimal for batched

#define NUM_THREADS 128 //64 //128

#define BLOCK_SIZE_N 128
#define DIM_X_N 128
#define DIM_Y_N 1

#define BLOCK_SIZE_T 32
#define DIM_X_T 16
#define DIM_Y_T 8


#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

extern __shared__ float shared_data[];


template<magma_trans_t transA, magma_diag_t diag>
static __device__ void
strsv_backwards_tri_device(int n,
    const float* __restrict__ A, int lda,
    float* __restrict__ b, int incb,
    float* sx)

{
    /*
    assume sx is in shared memory
    */
    int tx = threadIdx.x;
    float a;

    for (int step = 0; step < n; step++)
    {
        if (tx < n)
        {
            if (transA == MagmaNoTrans)
            {
                a = A[(n - 1) + (n - 1) * lda - tx - step * lda]; // rowwise access data in a coalesced way
            }
            else if (transA == MagmaTrans)
            {
                a = A[(n - 1) + (n - 1) * lda - tx * lda - step]; // columwise access data, not in a coalesced way
            }
            else
            {
                a = MAGMA_S_CONJ(A[(n - 1) + (n - 1) * lda - tx * lda - step]); // columwise access data, not in a coalesced way
            }


            if (tx == step)
            {
                if (diag == MagmaUnit)
                {
                    sx[n - 1 - tx] = (b[n - 1 - tx] - sx[n - 1 - tx]);
                }
                else
                {
                    sx[n - 1 - tx] = (b[n - 1 - tx] - sx[n - 1 - tx]) / a;
                }
            }
        }
        __syncthreads(); // there should be a sych here but can be avoided if BLOCK_SIZE =32

        if (tx < n)
        {
            if (tx > step)
            {
                sx[n - 1 - tx] += a * sx[n - 1 - step];
            }
        }
    }
}


#define make_FloatingPoint(x, y) (x)

template< int n, typename T >
__device__ void
magma_sum_reduce( /*int n,*/ int i, T* x)
{
    __syncthreads();
    if (n > 1024) { if (i < 1024 && i + 1024 < n) { x[i] += x[i + 1024]; }  __syncthreads(); }
    if (n > 512) { if (i < 512 && i + 512 < n) { x[i] += x[i + 512]; }  __syncthreads(); }
    if (n > 256) { if (i < 256 && i + 256 < n) { x[i] += x[i + 256]; }  __syncthreads(); }
    if (n > 128) { if (i < 128 && i + 128 < n) { x[i] += x[i + 128]; }  __syncthreads(); }
    if (n > 64) { if (i < 64 && i + 64 < n) { x[i] += x[i + 64]; }  __syncthreads(); }
    if (n > 32) { if (i < 32 && i + 32 < n) { x[i] += x[i + 32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if (n > 16) { if (i < 16 && i + 16 < n) { x[i] += x[i + 16]; }  __syncthreads(); }
    if (n > 8) { if (i < 8 && i + 8 < n) { x[i] += x[i + 8]; }  __syncthreads(); }
    if (n > 4) { if (i < 4 && i + 4 < n) { x[i] += x[i + 4]; }  __syncthreads(); }
    if (n > 2) { if (i < 2 && i + 2 < n) { x[i] += x[i + 2]; }  __syncthreads(); }
    if (n > 1) { if (i < 1 && i + 1 < n) { x[i] += x[i + 1]; }  __syncthreads(); }
}
// end sum_reduce

template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
static __device__ void
gemvn_template_device(
    int m, int n, T alpha,
    const T* __restrict__ A, int lda,
    const T* __restrict__ x, int incx, T beta,
    T* __restrict__ y, int incy)
{
    if (m <= 0 || n <= 0) return;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;

    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int tx = thread_id % DIM_X;
    int ty = thread_id / DIM_X;

    int ind = blockIdx.x * TILE_SIZE + tx;

    __shared__ T sdata[DIM_X * DIM_Y];


    int st = blockIdx.x * TILE_SIZE;

    int ed = min(st + TILE_SIZE, magma_roundup(m, DIM_X));

    int iters = (ed - st) / DIM_X;

    for (int i = 0; i < iters; i++)
    {
        if (ind < m) A += ind;

        T res = make_FloatingPoint(0.0, 0.0);

        if (ind < m)
        {
            for (int col = ty; col < n; col += DIM_Y)
            {
                res += A[col * lda] * x[col * incx];
            }
        }

        if (DIM_X >= num_threads) // indicated 1D threads configuration. Shared memory is not needed, reduction is done naturally
        {
            if (ty == 0 && ind < m)
            {
                y[ind * incy] = alpha * res + beta * y[ind * incy];
            }
        }
        else
        {
            sdata[ty + tx * DIM_Y] = res;

            __syncthreads();

            if (DIM_Y > 16)
            {
                magma_sum_reduce< DIM_Y >(ty, sdata + tx * DIM_Y);
            }
            else
            {
                if (ty == 0 && ind < m)
                {
                    for (int i = 1; i < DIM_Y; i++)
                    {
                        sdata[tx * DIM_Y] += sdata[i + tx * DIM_Y];
                    }
                }
            }

            if (ty == 0 && ind < m)
            {
                y[ind * incy] = alpha * sdata[tx * DIM_Y] + beta * y[ind * incy];
            }

            __syncthreads();
        }

        if (ind < m) A -= ind;

        ind += DIM_X;
    }
}


/******************************************************************************/
/*
    used in lower nontranspose and upper transpose
*/
template<magma_trans_t transA, magma_diag_t diag>
static __device__ void
strsv_forwards_tri_device(int n,
    const float* __restrict__ A, int lda,
    float* __restrict__ b, int incb,
    float* sx)

{
    /*
    assume sx is in shared memory
    */
    int tx = threadIdx.x;
    float a;

    for (int step = 0; step < n; step++)
    {
        if (tx < n) // hard code to BLOCK_SIZE and test divisible case only make 1Gflop/s difference
        {
            if (transA == MagmaNoTrans)
            {
                a = A[tx + step * lda]; // rowwise access data in a coalesced way
            }
            else  if (transA == MagmaTrans)
            {
                a = A[tx * lda + step]; // columwise access data, not in a coalesced way
            }
            else
            {
                a = MAGMA_S_CONJ(A[tx * lda + step]); // columwise access data, not in a coalesced way
            }


            if (tx == step)
            {
                if (diag == MagmaUnit)
                {
                    sx[tx] = (b[tx] - sx[tx]);
                }
                else
                {
                    sx[tx] = (b[tx] - sx[tx]) / a;
                }
            }
        }
        __syncthreads(); // there should be a sych here but can be avoided if BLOCK_SIZE =32

        if (tx < n)
        {
            if (tx > step)
            {
                sx[tx] += a * sx[step];
            }
        }
    }
}

template<const int BLOCK_SIZE, const int BLK_X, const int BLK_Y, const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
static __device__ void
strsv_notrans_device(
    int n,
    const float* __restrict__ A, int lda,
    float* b, int incb,
    float* x)
{
    int tx = threadIdx.x;
    int col = n;
    float* sx = (float*)shared_data;

    if (flag == 0)
    {
        for (int j = tx; j < n; j += BLOCK_SIZE)
        {
            sx[j] = MAGMA_S_ZERO;
        }
    }
    else
    {
        for (int j = tx; j < n; j += BLOCK_SIZE)
        {
            sx[j] = x[j];
        }
    }
    __syncthreads();


    if (uplo == MagmaUpper)
    {
        for (int i = 0; i < n; i += BLOCK_SIZE)
        {
            int jb = min(BLOCK_SIZE, n - i);
            col -= jb;

            gemvn_template_device<float, BLK_X, BLK_Y, TILE_SIZE>(jb, i, MAGMA_S_ONE, A(col, col + jb), lda, sx + col + jb, 1, MAGMA_S_ONE, sx + col, 1);
            __syncthreads();

            strsv_backwards_tri_device<trans, diag>(jb, A(col, col), lda, b + col, incb, sx + col);
            __syncthreads();
        }
    }
    else
    {
        for (int i = 0; i < n; i += BLOCK_SIZE)
        {
            int jb = min(BLOCK_SIZE, n - i);
            col = i;

            gemvn_template_device<float, BLK_X, BLK_Y, TILE_SIZE>(jb, i, MAGMA_S_ONE, A(col, 0), lda, sx, 1, MAGMA_S_ONE, sx + col, 1);
            __syncthreads();

            strsv_forwards_tri_device<trans, diag>(jb, A(col, col), lda, b + col, incb, sx + col);
            __syncthreads();
        }
    }


    for (int j = tx; j < n; j += BLOCK_SIZE)
    {
        x[j] = sx[j]; // write to x in reverse order
    }
    __syncthreads();
}


/******************************************************************************/
template< const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
__global__ void
strsv_notrans_kernel_outplace_batched(
    int n,
    float **A_array, int lda,
    float **b_array, int incb,
    float **x_array)
{
    int batchid = blockIdx.z;

    strsv_notrans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>(n, A_array[batchid], lda, b_array[batchid], incb, x_array[batchid]);
}


/******************************************************************************/
/*template<const int BLOCK_SIZE, const int DIM_X, const int DIM_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag>
__global__ void
strsv_trans_kernel_outplace_batched(
    int n,
    float **A_array, int lda,
    float **b_array, int incb,
    float **x_array)
{
    int batchid = blockIdx.z;
    strsv_trans_device<BLOCK_SIZE, DIM_X, DIM_Y, TILE_SIZE, flag, uplo, trans, diag>(n, A_array[batchid], lda, b_array[batchid], incb, x_array[batchid]);
}*/



/******************************************************************************/
extern "C" void
magmablas_strsv_outofplace_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    float ** A_array, magma_int_t lda,
    float **b_array, magma_int_t incb,
    float **x_array,
    magma_int_t batchCount, cudaStream_t queue,
    magma_int_t flag)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    
    // quick return if possible.
    if (n == 0)
        return;

    dim3 threads( NUM_THREADS, 1, 1 );
    dim3 blocks( 1, 1, batchCount );
    size_t shmem = n * sizeof(float);

    if (trans == MagmaNoTrans)
    {
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit)
            {
                if (flag == 0) {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaUpper, MagmaNoTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaUpper, MagmaNoTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit)
            {
                if (flag == 0) {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaUpper, MagmaNoTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaUpper, MagmaNoTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
        else //Lower
        {
            if (diag == MagmaNonUnit)
            {
                if (flag == 0)
                {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaLower, MagmaNoTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaLower, MagmaNoTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit)
            {
                if (flag == 0)
                {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 0, MagmaLower, MagmaNoTrans, MagmaUnit>
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_notrans_kernel_outplace_batched< BLOCK_SIZE_N, DIM_X_N, DIM_Y_N, MagmaBigTileSize, 1, MagmaLower, MagmaNoTrans, MagmaUnit>
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
    }
    else if (trans == MagmaTrans)
    {
        /*
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
        else
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
        */
    }
    else if (trans == MagmaConjTrans)
    {
        /*
        if (uplo == MagmaUpper)
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaConjTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaConjTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 0, MagmaUpper, MagmaConjTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaUpper, MagmaConjTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
        else
        {
            if (diag == MagmaNonUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaConjTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaConjTrans, MagmaNonUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
            else if (diag == MagmaUnit) {
                if (flag == 0)
                {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T,MagmaBigTileSize, 0, MagmaLower, MagmaConjTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
                else {
                    strsv_trans_kernel_outplace_batched< BLOCK_SIZE_T, DIM_X_T, DIM_Y_T, MagmaBigTileSize, 1, MagmaLower, MagmaConjTrans, MagmaUnit >
                        <<< blocks, threads, shmem, queue  >>>
                        (n, A_array, lda, b_array, incb, x_array);
                }
            }
        }
        */
    }
}


/******************************************************************************/
/*
extern "C" void
magmablas_strsv_recursive_outofplace_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    float **A_array, magma_int_t lda,
    float **b_array, magma_int_t incb,
    float **x_array,
    magma_int_t batchCount, cudaStream_t queue)
{
    /* Check arguments */
/*
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;


    //Init x_array with zero
    //magmablas_slaset_batched(MagmaFull, n, incb, MAGMA_S_ZERO, MAGMA_S_ZERO, x_array, n, batchCount, queue);
   
    //memory allocation takes 0.32ms

    float **dW0_displ  = NULL;
    float **dW1_displ  = NULL;
    float **dW2_displ  = NULL;

    magma_int_t alloc = 0;

    alloc += magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    alloc += magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    alloc += magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));

    if (alloc != 0)
    {
        magma_free( dW0_displ );
        magma_free( dW1_displ );
        magma_free( dW2_displ );

        info = MAGMA_ERR_DEVICE_ALLOC;
        return;
    }

    magma_int_t col = n;

    if (trans == MagmaNoTrans)
    {
        for (magma_int_t i=0; i < n; i+= NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaUpper)
            {
                col -= jb;

                magma_sdisplace_pointers(dW0_displ, A_array, lda, col, col+jb, batchCount, queue);
                magma_sdisplace_pointers(dW1_displ, x_array, 1, col+jb, 0,     batchCount, queue);
                magma_sdisplace_pointers(dW2_displ, x_array, 1, col,    0,     batchCount, queue);
            }
            else
            {
                col = i;
                
                magma_sdisplace_pointers(dW0_displ, A_array, lda, col, 0, batchCount, queue);
                magma_sdisplace_pointers(dW1_displ, x_array, 1,   0,   0, batchCount, queue);
                magma_sdisplace_pointers(dW2_displ, x_array, 1,   col, 0, batchCount, queue);
            }
            
            //assume x_array contains zero elements
            magmablas_sgemv_batched(MagmaNoTrans, jb, i, MAGMA_S_ONE, dW0_displ, lda, dW1_displ, 1, MAGMA_S_ONE, dW2_displ, 1, batchCount, queue);
            
            magma_sdisplace_pointers(dW0_displ, A_array, lda,  col, col, batchCount, queue);
            magma_sdisplace_pointers(dW1_displ, b_array, 1, col*incb,   0, batchCount, queue);
            magma_sdisplace_pointers(dW2_displ, x_array, 1,    col,   0, batchCount, queue);
            
            magmablas_strsv_outofplace_batched(uplo, trans, diag,jb, dW0_displ, lda, dW1_displ, incb, dW2_displ, batchCount, queue, i);
        }
    }
    else
    {
        for (magma_int_t i=0; i < n; i += NB)
        {
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaLower)
            {
                col -= jb;

                magma_sdisplace_pointers(dW0_displ, A_array, lda, col+jb, col, batchCount, queue);
                magma_sdisplace_pointers(dW1_displ, x_array, 1, col+jb, 0,     batchCount, queue);
                magma_sdisplace_pointers(dW2_displ, x_array, 1, col,    0,     batchCount, queue);
            }
            else
            {
                col = i;
                
                magma_sdisplace_pointers(dW0_displ, A_array, lda, 0, col,  batchCount, queue);
                magma_sdisplace_pointers(dW1_displ, x_array, 1,   0,   0, batchCount, queue);
                magma_sdisplace_pointers(dW2_displ, x_array, 1,   col, 0, batchCount, queue);
            }


            //assume x_array contains zero elements
            
            magmablas_sgemv_batched(trans, i, jb, MAGMA_S_ONE, dW0_displ, lda, dW1_displ, 1, MAGMA_S_ONE, dW2_displ, 1, batchCount, queue);
            
            magma_sdisplace_pointers(dW0_displ, A_array, lda,  col, col, batchCount, queue);
            magma_sdisplace_pointers(dW1_displ, b_array, 1, col*incb,   0, batchCount, queue);
            magma_sdisplace_pointers(dW2_displ, x_array, 1,    col,   0, batchCount, queue);
            
            magmablas_strsv_outofplace_batched(uplo, trans, diag, jb, dW0_displ, lda, dW1_displ, incb, dW2_displ, batchCount, queue, i);
        }
    }

    magma_free(dW0_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
}

*/



/*****************************************************************************
extern "C" void
magmablas_strsv_work_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    float **A_array, magma_int_t lda,
    float **b_array, magma_int_t incb,
    float **x_array,
    magma_int_t batchCount, cudaStream_t queue)
{
    //magmablas_slaset_batched(MagmaFull, n, incb, MAGMA_S_ZERO, MAGMA_S_ZERO, x_array, n, batchCount, queue);

    //magmablas_strsv_recursive_outofplace_batched

    magmablas_strsv_recursive_outofplace_batched(uplo, trans, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue);

    magmablas_slacpy_batched( MagmaFull, n, incb, x_array, n, b_array, n, batchCount, queue);
}*/


/***************************************************************************//**
    Purpose
    -------
    strsv solves one of the matrix equations on gpu

        op(A)*x = b,   or
        x*op(A) = b,

    where alpha is a scalar, X and B are vectors, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The vector x is overwritten on b.


    Arguments
    ----------

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    trans  magma_trans_t.
            On entry, trans specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n N specifies the order of the matrix A. n >= 0.

    @param[in]
    A_array       Array of pointers, dimension (batchCount).
            Each is a REAL array A of dimension ( lda, n ),
            Before entry with uplo = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    b_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension  n
            On exit, b is overwritten with the solution vector X.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trsv_batched
*******************************************************************************/
/*
extern "C" void
magmablas_strsv_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    float **A_array, magma_int_t lda,
    float **b_array, magma_int_t incb,
    magma_int_t batchCount,
    cudaStream_t queue)
{
    magma_int_t size_x = n * incb;

    float *x=NULL;
    float **x_array = NULL;

    magma_smalloc( &x, size_x * batchCount);
    magma_malloc((void**)&x_array,  batchCount * sizeof(*x_array));

    magma_sset_pointer( x_array, x, n, 0, 0, size_x, batchCount, queue );

    magmablas_strsv_work_batched(uplo, trans, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue);

    magma_free(x);
    magma_free(x_array);
}
*/