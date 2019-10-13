

#include "utils.h"
#include "magma_types.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__global__ void zdisplace_pointers_kernel(float** output_array,
    float** input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column)
{
    float* inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
}

extern "C"
void magma_sdisplace_pointers(float** output_array,
    float** input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, cudaStream_t queue)
{
    zdisplace_pointers_kernel
        << < batchCount, 1, 0, queue >> >
        (output_array, input_array, lda, row, column);
}


static
__global__ void magma_iset_pointer_kernel(
    magma_int_t** output_array,
    magma_int_t* input,
    int lda,
    int row, int column,
    int batchSize)
{
    output_array[blockIdx.x] = input + blockIdx.x * batchSize + row + column * lda;
}

extern "C"
void magma_iset_pointer(
    magma_int_t * *output_array,
    magma_int_t * input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchSize,
    magma_int_t batchCount, cudaStream_t queue)
{
    /*
    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batchSize)
    change into A_array[0] A_array[1],... A_array[batchCount-1], where the size of each A_array[i] is N
    */
    magma_iset_pointer_kernel
        << < batchCount, 1, 0, queue >> >
        (output_array, input, lda, row, column, batchSize);
}



__global__ void kernel_sset_pointer(
    float** output_array,
    float* input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batch_offset)
{
    output_array[blockIdx.x] = input + blockIdx.x * batch_offset + row + column * lda;
    //printf("==> kernel_set_pointer input_array %p output_array %p  \n",input+ blockIdx.x * batch_offset,output_array[blockIdx.x]);
}

extern "C"
void magma_sset_pointer(
    float** output_array,
    float* input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batch_offset,
    magma_int_t batchCount,
    cudaStream_t queue)
{
    kernel_sset_pointer
        << < batchCount, 1, 0, queue >> >
        (output_array, input, lda, row, column, batch_offset);
}




// To deal with really large matrices, this launchs multiple super blocks,
// each with up to 64K-1 x 64K-1 thread blocks, which is up to 4194240 x 4194240 matrix with BLK=64.
// CUDA architecture 2.0 limits each grid dimension to 64K-1.
// Instances arose for vectors used by sparse matrices with M > 4194240, though N is small.
const magma_int_t max_blocks = 65535;

// BLK_X and BLK_Y need to be equal for slaset_q to deal with diag & offdiag
// when looping over super blocks.
// Formerly, BLK_X and BLK_Y could be different.
#define BLK_X 64
#define BLK_Y BLK_X

/*
    kernel wrappers to call the device functions.
*/

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to slaset, slacpy, slag2d, clag2z, sgeadd.
*/
static __device__
void slaset_full_device(
    int m, int n,
    float offdiag, float diag,
    float* A, int lda)
{
    int ind = blockIdx.x * BLK_X + threadIdx.x;
    int iby = blockIdx.y * BLK_Y;
    /* check if full block-column && (below diag || above diag || offdiag == diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y || ind + BLK_X <= iby || MAGMA_S_EQUAL(offdiag, diag)));
    /* do only rows inside matrix */
    if (ind < m) {
        A += ind + iby * lda;
        if (full) {
            // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
            for (int j = 0; j < BLK_Y; ++j) {
                A[j * lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for (int j = 0; j < BLK_Y && iby + j < n; ++j) {
                if (iby + j == ind)
                    A[j * lda] = diag;
                else
                    A[j * lda] = offdiag;
            }
        }
    }
}


__global__
void slaset_full_kernel(
    int m, int n,
    float offdiag, float diag,
    float* dA, int ldda)
{
    slaset_full_device(m, n, offdiag, diag, dA, ldda);
}

extern "C"
void magmablas_slaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda,
    cudaStream_t queue)
{
#define dA(i_, j_) (dA + (i_) + (j_)*ldda)

    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
        info = -1;
    else if (m < 0)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (ldda < max(1, m))
        info = -7;

    if (info != 0) {
        magma_xerbla(__func__, -(info));
        return;  //info;
    }

    if (m == 0 || n == 0) {
        return;
    }

    const magma_int_t super_NB = max_blocks * BLK_X;
    dim3 super_grid(magma_ceildiv(m, super_NB), magma_ceildiv(n, super_NB));

    dim3 threads(BLK_X, 1);
    dim3 grid;

    magma_int_t mm, nn;


    // if continuous in memory & set to zero, cudaMemset is faster.
    // TODO: use cudaMemset2D ?
    if (m == ldda &&
        MAGMA_S_EQUAL(offdiag, MAGMA_S_ZERO) &&
        MAGMA_S_EQUAL(diag, MAGMA_S_ZERO))
    {
        size_t size = m * n;
        cudaError_t err = cudaMemsetAsync(dA, 0, size * sizeof(float), queue);
        assert(err == cudaSuccess);
        MAGMA_UNUSED(err);
    }
    else {
        for (unsigned int i = 0; i < super_grid.x; ++i) {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = magma_ceildiv(mm, BLK_X);
            for (unsigned int j = 0; j < super_grid.y; ++j) {  // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = magma_ceildiv(nn, BLK_Y);
                if (i == j) {  // diagonal super block
                    slaset_full_kernel << < grid, threads, 0, queue >> >
                        (mm, nn, offdiag, diag, dA(i * super_NB, j * super_NB), ldda);
                }
                else {           // off diagonal super block
                    slaset_full_kernel << < grid, threads, 0, queue >> >
                        (mm, nn, offdiag, offdiag, dA(i * super_NB, j * super_NB), ldda);
                }
            }
        }
    }
}
