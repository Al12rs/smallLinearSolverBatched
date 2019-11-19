

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <lapacke.h>
//#include <cusolverDn.h>
#include "utils.h"
#include "operation_batched.h"
#include "testings.h"
#include "flops.h"
#include "cuda_profiler_api.h"



#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h" // internal header
#endif

int gpuLinearSolverBatched_tester(int N, int batchCount);

int main(int argc, char **argv)
{
    int N, batchCount;
    if (argc != 3){
        printf("Usage: linearSolverBatched <linear system size> <number of systems>\n");
        return 0;
    }
    else {
        N = atoi(argv[1]);
        batchCount = atoi(argv[2]);
        printf("Performing Solve test with N=%d, and batchCount=%d\n", N, batchCount);
        //test to see if cublas initialization is slowing down later
        cublasHandle_t handle;
        cublasCreate(&handle);
        gpuLinearSolverBatched_tester(N, batchCount);
        cublasDestroy(handle);
        return 0;
    }
}

int gpuLinearSolverBatched_tester(int N, int batchCount)
{
    int sizeA, sizeB, result;
    float *h_A, *h_B, *h_X;
    int *h_info;

    sizeA = N * N * batchCount;
    sizeB = N * 1 * batchCount;

    //Allocate A and B matrices
    TESTING_CHECK(magma_smalloc_cpu(&h_A, sizeA));
    TESTING_CHECK(magma_smalloc_cpu(&h_B, sizeB));
    TESTING_CHECK(magma_smalloc_cpu(&h_X, sizeB));
    TESTING_CHECK(magma_imalloc_cpu(&h_info, batchCount));

    curandGenerator_t hostRandGenerator;
    curandCreateGeneratorHost(&hostRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);

    /* Initialize the matrices */
    curandGenerateNormal(hostRandGenerator, h_A, sizeA, (float)0, (float)1);
    curandGenerateNormal(hostRandGenerator, h_B, sizeB, (float)0, (float)1);

    //First run test, this seems slower, possibly due to library loading
    clock_t begin = clock();
    cudaProfilerStart();
    result = gpuLinearSolverBatched(N, h_A, h_B, &h_X, h_info, batchCount);
    cudaProfilerStop();
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Batched Solve operation finished with exit code: %d in %f\n", result, time_spent);

    //Second run test, this seems to perform much better, needs investigation
    begin = clock();
    cudaProfilerStart();
    result = gpuLinearSolverBatched(N, h_A, h_B, &h_X, h_info, batchCount);
    cudaProfilerStop();
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Batched Solve operation finished with exit code: %d in %f\n", result, time_spent);

#define LAPACK_PERFORMANCE
#ifdef LAPACK_PERFORMANCE
    /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */

    double cpu_time;
    int nrhs = 1;
    int lda = N;
    int ldb = N;
    int *ipiv;
    TESTING_CHECK(magma_imalloc_cpu(&ipiv, batchCount * N));
// #define BATCHED_DISABLE_PARCPU
    cpu_time = clock();
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (magma_int_t s = 0; s < batchCount; s++)
    {
        magma_int_t locinfo;
        sgesv_(&N, &nrhs, h_A + s * lda * N, &lda, ipiv + s * N, h_B + s * ldb * nrhs, &ldb, &locinfo);
        if (locinfo != 0)
        {
            printf("lapackf77_sgesv matrix %lld returned error %lld: %s.\n",
                   (long long)s, (long long)locinfo, "magma_strerror(locinfo)");
        }
    }
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);
#endif

    cpu_time = (double)(clock() - cpu_time)/CLOCKS_PER_SEC;
    printf("lapack computation finished in: %f\n", cpu_time);
    //cpu_perf = gflops / cpu_time;

    // #define BATCHED_DISABLE_PARCPU
    cpu_time = clock();
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    magma_int_t nthreads = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);
    magma_set_omp_numthreads(nthreads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (magma_int_t s = 0; s < batchCount; s++)
    {
        magma_int_t locinfo;
        sgesv_(&N, &nrhs, h_A + s * lda * N, &lda, ipiv + s * N, h_B + s * ldb * nrhs, &ldb, &locinfo);
        if (locinfo != 0)
        {
            printf("lapackf77_sgesv matrix %lld returned error %lld: %s.\n",
                   (long long)s, (long long)locinfo, "magma_strerror(locinfo)");
        }
    }
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    magma_set_lapack_numthreads(nthreads);
#endif

    cpu_time = (double)(clock() - cpu_time) / CLOCKS_PER_SEC;
    printf("lapack computation finished in: %f\n", cpu_time);
    //cpu_perf = gflops / cpu_time;
    #endif //LAPACK_PERFORMANCE

    magma_free_cpu(h_A);
    magma_free_cpu(h_B);
    magma_free_cpu(h_X);
    magma_free_cpu(h_info);
    return result;
}



