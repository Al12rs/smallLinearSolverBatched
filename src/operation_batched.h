#ifndef OPERATION_BATCHED_H
#define OPERATION_BATCHED_H

#include <cuda_runtime.h>
#include "magma_types.h"

#if __cplusplus
extern "C" {
#endif

    int linearDecompSLU_batched(
        int m, int n,
        float** dA_array,
        int ldda,
        int** ipiv_array, int* info_array,
        int batchCount, cudaStream_t queue);

    int linearSolverFactorizedSLU_batched(
        int n, int nrhs,
        float** dA_array, int ldda,
        int** dipiv_array,
        float** dB_array, int lddb,
        int batchCount, cudaStream_t queue);

    int linearSolverSLU_batched(int n, int nrhs,
        float** dA_array, int ldda,
        int** dipiv_array,
        float** dB_array, int lddb,
        int* dinfo_array,
        int batchCount, cudaStream_t queue);

    void magma_iset_pointer(
        magma_int_t** output_array,
        magma_int_t* input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column,
        magma_int_t batchSize,
        magma_int_t batchCount, cudaStream_t queue);

    void magma_sset_pointer(
        float** output_array,
        float* input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column,
        magma_int_t batch_offset,
        magma_int_t batchCount,
        cudaStream_t queue);
    
    //tinySLUfactorization_batched.cu
    
    magma_int_t magma_sgetrf_batched_smallsq_noshfl( 
        magma_int_t n, 
        float** dA_array,
        magma_int_t ldda, 
        magma_int_t** ipiv_array,
        magma_int_t* info_array, 
        magma_int_t batchCount, 
        cudaStream_t queue );

    magma_int_t magma_sgetrf_batched_smallsq_shfl(
        magma_int_t n,
        float** dA_array,
        magma_int_t ldda,
        magma_int_t** ipiv_array,
        magma_int_t* info_array,
        magma_int_t batchCount,
        cudaStream_t queue);

    //linearSolver(Alexpart).cu

    void magma_slaswp_rowserial_batched(
        magma_int_t n, 
        float** dA_array, 
        magma_int_t lda,
        magma_int_t k1,
        magma_int_t k2,
        magma_int_t** ipiv_array,
        magma_int_t batchCount, 
        cudaStream_t queue);

    void magmablas_strsv_outofplace_batched(
        magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
        magma_int_t n,
        float** A_array, magma_int_t lda,
        float** b_array, magma_int_t incb,
        float** x_array,
        magma_int_t batchCount, cudaStream_t queue,
        magma_int_t flag);

    /*void magmablas_sgemv_batched(
        magma_trans_t trans, magma_int_t m, magma_int_t n,
        float alpha,
        magmaFloat_ptr dA_array[], magma_int_t ldda,
        magmaFloat_ptr dx_array[], magma_int_t incx,
        float beta,
        magmaFloat_ptr dy_array[], magma_int_t incy,
        magma_int_t batchCount, cudaStream_t queue);*/

    void magmablas_slaset(
        magma_uplo_t uplo, magma_int_t m, magma_int_t n,
        float offdiag, float diag,
        magmaFloat_ptr dA, magma_int_t ldda,
        cudaStream_t queue);

    /*void magma_sdisplace_pointers(float** output_array,
        float** input_array, magma_int_t lda,
        magma_int_t row, magma_int_t column,
        magma_int_t batchCount, cudaStream_t queue);*/

#if __cplusplus
}
#endif

#endif //OPERATION_BATCHED_H