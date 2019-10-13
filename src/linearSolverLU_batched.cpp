#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <cublas_v2.h>
#include "utils.h"
#include "operation_batched.h"


#define NOTRANSF 111

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/***************************************************************************//**
    Purpose
    -------
    Solves a system of linear equations
       A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as
       A = P * L * U,
    where P is a permutation matrix, L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    This is a batched version that solves batchCount N-by-N matrices in parallel.
    dA, dB, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

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
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).


    @param[in,out]
    dB_array   Array of pointers, dimension (batchCount).
            Each is a REAL array on the GPU, dimension (LDDB,N).
            On entry, each pointer is an right hand side matrix B.
            On exit, each pointer is the solution matrix X.


    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).


    @param[out]
    dinfo_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
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
    queue   cudaStream_t
            Stream to execute in.

*******************************************************************************/
extern "C" int linearSolverSLU_batched(int n, int nrhs,
                                       float **dA_array, int ldda,
                                       int **dipiv_array,
                                       float **dB_array, int lddb,
                                       int * dinfo_array,
                                       int batchCount, cudaStream_t queue)
{                                      
    /* Local variables */
    int info;
    info = 0;
    if (n < 0) {
        info = -1;
    }
    else if (nrhs < 0) {
        info = -2;
    }
    else if (ldda < max(1, n)) {
        info = -4;
    }
    else if (lddb < max(1, n)) {
        info = -6;
    }
    if (info != 0) {
        utils_reportError(__func__, -(info));
        return info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return info;
    }
    info = linearDecompSLU_batched(n, n, dA_array, ldda, dipiv_array, dinfo_array, batchCount, queue);
    if (info != RES_SUCCESS) {
        return info;
    }

    // TODO: clean this
#define CHECK_INFO
#ifdef CHECK_INFO
    // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
    magma_int_t* cpu_info = NULL;
    magma_imalloc_cpu(&cpu_info, batchCount);
    cublasGetVectorAsync(
                    int(batchCount), sizeof(int),
                    dinfo_array, 1,
                    cpu_info, 1, NULL);
                cudaStreamSynchronize(NULL);
    //magma_getvector(batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
    for (magma_int_t i = 0; i < batchCount; i++)
    {
        if (cpu_info[i] != 0) {
            printf("magma_sgetrf_batched matrix %lld returned error %lld\n", (long long)i, (long long)cpu_info[i]);
            info = cpu_info[i];
            magma_free_cpu(cpu_info);
            return info;
        }
    }
    magma_free_cpu(cpu_info);
#endif
    
    info = linearSolverFactorizedSLU_batched( n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchCount, queue);
    return info;
}

#undef min
#undef max
