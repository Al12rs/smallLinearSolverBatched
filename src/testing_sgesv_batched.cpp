

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <lapacke.h>
#include <sys/time.h>
#include "utils.h"
#include "operation_batched.h"
#include "testings.h"
#include "flops.h"
#include "cuda_profiler_api.h"

// If MANUAL_TEST is defined, the porgram will perform a single test with command line parameters
// for matrix size, batch count and number of threads.
// If  MANUAL_TEST is not defined then the automatic tester will execute with the parameters set in gpuCSVTester().
#define MANUAL_TEST

// Defining DISABLE_GPU_TEST will avoid perormaing GPU solver when using manual mode. This option is ignored in automatic tester mode.
//#define DISABLE_GPU_TEST

// defining LAPACK_PERFORMANCE enables CPU test in manual mode, as it's normally skipped. This option does not work for
// automatic tester as there CPU performance is always tested.
//#define LAPACK_PERFORMANCE

// Defining BATCHED_DISABLE_PARCPU will disable OMP multithreading directives and block the use of multiple threads for CPU test.
//#define BATCHED_DISABLE_PARCPU
#if defined(_OPENMP)
#include <omp.h>
#endif

int gpuLinearSolverBatched_tester(int N, int batchCount, int numThreads);
int gpuCSVTester();

#ifdef MANUAL_TEST
int main(int argc, char **argv)
{
    int N, batchCount, numThreads;
    if (argc != 4)
    {
        printf("Usage: gpulinearsolversmallbatched <linear system size> <number of systems> <number of omp threads>\n");
        return 0;
    }
    else
    {
        N = atoi(argv[1]);
        batchCount = atoi(argv[2]);
        numThreads = atoi(argv[3]);
        int mem = ((N + 1) * N * batchCount * sizeof(float));
        printf("Performing Solve test: N=%d, batchCount=%d, estimated memory usage:%d\n", N, batchCount, mem);
        //test to see if cublas initialization is slowing down later

        cublasHandle_t handle;
        cublasCreate(&handle);
        gpuLinearSolverBatched_tester(N, batchCount, numThreads);
        cublasDestroy(handle);
        return 0;
    }
}
#else
int main(int argc, char **argv)
{
    cublasHandle_t handle;

    // Initialize Cublas even if not strictly necessary, but this should
    // avoids the Cuda runtime initialization overhead later,
    // during the first test execution.
    cublasCreate(&handle);

    //Tester
    gpuCSVTester();

    cublasDestroy(handle);
}
#endif

#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
int omp_thread_count()
{
    int n = 0;
#pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
#endif

// Autotester
int gpuCSVTester()
{
    int sizeA, sizeB, result;
    float *h_A, *h_B, *h_X;
    int *h_info;
    double time_spent;
    struct timeval t1, t2;
    long int clockCycles;
    int N, batchCount, memByte;
    double memMB;
    double cpuTime[5];
    FILE *fp;

    curandGenerator_t hostRandGenerator;
    curandCreateGeneratorHost(&hostRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);

    fp = fopen("results.csv", "w+");
    fprintf(fp, "#N, #batchCount, #result, #GPUtimeInMs, #CPUtime1InMs, #CPUtime4InMs, #CPUtime8InMs, #CPUtime16InMs, #memMB, #memByte\n");

    // Modify to test different batchCount values
    for (int i = 0; i < 8; i++)
    {
        switch (i)
        {
        case 0:
            batchCount = 1000;
            break;
        case 1:
            batchCount = 10 * 1000;
            break;
        case 2:
            batchCount = 50 * 1000;
            break;
        case 3:
            batchCount = 100 * 1000;
            break;
        case 4:
            batchCount = 250 * 1000;
            break;
        case 5:
            batchCount = 500 * 1000;
            break;
        case 6:
            batchCount = 750 * 1000;
            break;
        case 7:
            batchCount = 1000 * 1000;
            break;
        default:
            //Should never be here
            break;
        }

        //test for all sizes of matrices
        for (N = 1; N <= 32; N++)
        {

            // Estimate of device memory usage, we have to keep in mind the padding introduced with ldda and lddb.
            // Actual usage is higher because there are some other arrays and values allocated.
            int memByte1 = batchCount * (32 + 36) * sizeof(float);
            int memByte2 = 32 * N * batchCount * sizeof(float);
            memByte = memByte1 + memByte2;

            //Handle overflow case       2147483647
            if (memByte < 0 || memByte1 < 0 || memByte2 < 0 || memByte >= 2105032704)
            {
                fprintf(fp, "%d, %d, -1, -1, -1, -1, -1, -1, -1, %d\n", N, batchCount, memByte);
                printf("%d, %d, -1, -1, -1, -1, -1, -1, -1, %d\n", N, batchCount, memByte);
                continue;
            }
            memMB = (double)memByte / (1024 * 1024);

            sizeA = N * N * batchCount;
            sizeB = N * 1 * batchCount;

            //Allocate A and B matrices
            TESTING_CHECK(magma_smalloc_cpu(&h_A, sizeA));
            TESTING_CHECK(magma_smalloc_cpu(&h_B, sizeB));
            TESTING_CHECK(magma_smalloc_cpu(&h_X, sizeB));
            TESTING_CHECK(magma_imalloc_cpu(&h_info, batchCount));

            // Initialize the matrices
            curandGenerateNormal(hostRandGenerator, h_A, sizeA, (float)0, (float)1);
            curandGenerateNormal(hostRandGenerator, h_B, sizeB, (float)0, (float)1);

            //Perform test on GPU
            gettimeofday(&t1, 0);
            result = gpuLinearSolverBatched(N, h_A, h_B, &h_X, h_info, batchCount);
            gettimeofday(&t2, 0);
            double gpuTime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

            //Perform test on CPU
            int nrhs = 1;
            int lda = N;
            int ldb = N;
            int *ipiv;
            TESTING_CHECK(magma_imalloc_cpu(&ipiv, batchCount * N));

            //Single thread CPU test
            gettimeofday(&t1, 0);
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
            gettimeofday(&t2, 0);
            cpuTime[0] = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

            //Multithreaded CPU test
            for (int numThreads = 4; numThreads <= 16; numThreads = numThreads * 2)
            {
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                omp_set_num_threads(numThreads);
                omp_set_max_active_levels(2);
                int oldNumThreads = omp_thread_count();

                printf("OMP threads: %d\n", oldNumThreads);
#endif
                gettimeofday(&t1, 0);
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                //schedule(dynamic)
#pragma omp parallel for
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
                gettimeofday(&t2, 0);
                cpuTime[numThreads / 4] = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
            }

            //Print results
            fprintf(fp, "%d, %d, %d, %f, %f, %f, %f, %f, %f, %d\n",
                    N, batchCount, result, gpuTime, cpuTime[0], cpuTime[1], cpuTime[2], cpuTime[4], memMB, memByte);
            printf("%d, %d, %d, %f, %f, %f, %f, %f, %f, %d\n",
                   N, batchCount, result, gpuTime, cpuTime[0], cpuTime[1], cpuTime[2], cpuTime[4], memMB, memByte);

            // Cleanup
            magma_free_cpu(h_A);
            magma_free_cpu(h_B);
            magma_free_cpu(h_X);
            magma_free_cpu(h_info);
        }
    }

    fclose(fp);
}

// Manual tester
int gpuLinearSolverBatched_tester(int N, int batchCount, int numThreads)
{
    int sizeA, sizeB, result;
    float *h_A, *h_B, *h_X;
    int *h_info;
    struct timeval t1, t2;
    int oldNumThreads;

    sizeA = N * N * batchCount;
    sizeB = N * 1 * batchCount;

    //Allocate A and B matrices
    TESTING_CHECK(magma_smalloc_cpu(&h_A, sizeA));
    TESTING_CHECK(magma_smalloc_cpu(&h_B, sizeB));
    TESTING_CHECK(magma_smalloc_cpu(&h_X, sizeB));
    TESTING_CHECK(magma_imalloc_cpu(&h_info, batchCount));

    curandGenerator_t hostRandGenerator;
    curandCreateGeneratorHost(&hostRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);

    // Initialize the matrices
    curandGenerateNormal(hostRandGenerator, h_A, sizeA, (float)0, (float)1);
    curandGenerateNormal(hostRandGenerator, h_B, sizeB, (float)0, (float)1);

//Perform test on GPU
#if !defined(DISABLE_GPU_TEST)

    //First run test, this seems slower, possibly due to library loading
    result = gpuLinearSolverBatched(N, h_A, h_B, &h_X, h_info, batchCount);

    gettimeofday(&t2, 0);

    double gpuTime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("Batched Solve operation finished with exit code: %d in %f\n", result, gpuTime);

    //Second run test, this seems to perform much better, needs investigation
    gettimeofday(&t1, 0);
    result = gpuLinearSolverBatched(N, h_A, h_B, &h_X, h_info, batchCount);
    gettimeofday(&t2, 0);

    gpuTime = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Batched Solve operation finished with exit code: %d in %f\n", result, gpuTime);

#endif //GPU_TEST

    //CPU test
#ifdef LAPACK_PERFORMANCE
    double cpu_time;
    int nrhs = 1;
    int lda = N;
    int ldb = N;
    int *ipiv;
    TESTING_CHECK(magma_imalloc_cpu(&ipiv, batchCount * N));

    gettimeofday(&t1, 0);
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
    gettimeofday(&t2, 0);

    cpu_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("lapack single thread finished in: %f\n", cpu_time);

#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    omp_set_num_threads(numThreads);
    omp_set_max_active_levels(2);
    oldNumThreads = omp_thread_count();

    printf("OMP threads: %d\n", oldNumThreads);
#endif

    gettimeofday(&t1, 0);

#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
//schedule(dynamic)
#pragma omp parallel for
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
    gettimeofday(&t2, 0);

    cpu_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("lapack computation finished in: %f\n", cpu_time);

//Second run with schedule dynamic
#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
    omp_set_num_threads(numThreads);
    omp_set_max_active_levels(2);
    oldNumThreads = omp_thread_count();

    printf("OMP threads: %d\n", oldNumThreads);
#endif

    gettimeofday(&t1, 0);

#if !defined(BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
//schedule(dynamic)
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
    gettimeofday(&t2, 0);

    cpu_time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("lapack dynamic ver finished in: %f\n", cpu_time);

#endif //LAPACK_PERFORMANCE

    magma_free_cpu(h_A);
    magma_free_cpu(h_B);
    magma_free_cpu(h_X);
    magma_free_cpu(h_info);
    return result;
}
