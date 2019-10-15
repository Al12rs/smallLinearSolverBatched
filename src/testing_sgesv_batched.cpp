


#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

//#pragma comment(lib,"cublas.lib")
//#pragma comment(lib,"curand.lib")


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
//#include <cusolverDn.h>
#include "utils.h"
#include "operation_batched.h"
#include "testings.h"
#include "flops.h"


/*
   -- MAGMA (version 2.5.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date August 2019

   @author Mark gates
   @author Azzam Haidar
   @author Tingxing Dong

   @generated from testing/testing_zgesv_batched.cpp, normal z -> s, Fri Aug  2 17:10:12 2019
 */
// includes, system

#if(0)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#endif



int main(int argc, char **argv)
{
    int N, batchCount, sizeA, sizeB, result;
    float *h_A, *h_B, *h_X;
    
    N = 2;
    batchCount = 1000;

    sizeA = N * N * batchCount;
	sizeB = N * 1 * batchCount;

    //Allocate A and B matrices
    TESTING_CHECK( magma_smalloc_cpu( &h_A, sizeA ));
    TESTING_CHECK( magma_smalloc_cpu( &h_B, sizeB ));
    TESTING_CHECK( magma_smalloc_cpu( &h_X, sizeB ));

    curandGenerator_t hostRandGenerator;
    curandCreateGeneratorHost(&hostRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);

    /* Initialize the matrices */
    curandGenerateUniform(hostRandGenerator, h_A, sizeA);
    curandGenerateUniform(hostRandGenerator, h_B, sizeB);

    result = gpuLinearSolverBatched(N, &h_A, &h_B, &h_X, batchCount);
    printf("Batched Solveoperation finished with exit code: %f", result);
	return result;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgesv_batched
*/
int mainOld(int argc, char **argv)
{
    magma_init();
    //magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, Rnorm, Anorm, Xnorm, *work;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_B, *h_X;
    magmaFloat_ptr d_A, d_B;
    magma_int_t *dipiv, *dinfo_array;
    magma_int_t *ipiv, *cpu_info;
    magma_int_t N, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    float **dA_array = NULL;
    float **dB_array = NULL;
    magma_int_t     **dipiv_array = NULL;

    //magma_opts opts( MagmaOptsBatched );
    //opts.parse_opts( argc, argv );
    
    //float tol = opts.tolerance; //* lapackf77_slamch("E");

    nrhs = 1;
    batchCount = 1000000;

    curandGenerator_t hostRandGenerator;
    curandCreateGeneratorHost(&hostRandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    const int numStreams = 4;
    cudaStream_t cuda_stream[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&cuda_stream[i]);
    }
    

    printf("%% BatchCount   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < 1; ++itest ) {
        for( int iter = 0; iter < 1; ++iter ) {
            N = 2;
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, 32 );  // multiple of 32 by default
            lddb   = ldda;
            //gflops = ( FLOPS_SGETRF( N, N ) + FLOPS_SGETRS( N, nrhs ) ) * batchCount / 1e9;
            
            sizeA = lda*N*batchCount;
            sizeB = ldb*nrhs*batchCount;

            TESTING_CHECK( magma_smalloc_cpu( &h_A, sizeA ));
            TESTING_CHECK( magma_smalloc_cpu( &h_B, sizeB ));
            TESTING_CHECK( magma_smalloc_cpu( &h_X, sizeB ));
            TESTING_CHECK( magma_smalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, batchCount*N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            
            TESTING_CHECK( magma_smalloc( &d_A, ldda*N*batchCount    ));
            TESTING_CHECK( magma_smalloc( &d_B, lddb*nrhs*batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv, N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc( (void**) &dB_array,    batchCount * sizeof(float*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrices */
            curandGenerateUniform(hostRandGenerator, h_A, sizeA);
            curandGenerateUniform(hostRandGenerator, h_B, sizeB);



            //lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            //slarnv_(&ione, ISEED, &sizeA, h_A);
            //lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            
            double startTime = magma_wtime();
            //copy matrix h_A on host to d_A on device cublasSetMatrixAsync + cudaStreamSynchronize
            cublasSetMatrixAsync(
                int(N), int(N*batchCount), sizeof(float),
                h_A, int(lda),
                d_A, int(ldda), cuda_stream[0]);
            //use a different stream to copy concurrently
            cublasSetMatrixAsync(
                int(N), int(nrhs * batchCount), sizeof(float),
                h_B, int(ldb),
                d_B, int(lddb), cuda_stream[1]);
            //missing error check

            //magma_ssetmatrix( N, N*batchCount,    h_A, lda, d_A, ldda, opts.queue );
            //magma_ssetmatrix( N, nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            //convert consecutive values into array of values with size ldda*N
            magma_iset_pointer(dipiv_array, dipiv, 1, 0, 0, N, batchCount, cuda_stream[2]);
            cudaStreamSynchronize(cuda_stream[0]);
            cudaStreamSynchronize(cuda_stream[1]);
            magma_sset_pointer( dA_array, d_A, ldda, 0, 0, ldda*N, batchCount, cuda_stream[0]);
            magma_sset_pointer( dB_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, cuda_stream[1]);

            // cudaStreamSynchronize( queue->cuda_stream() ); then there is the time for benchmarking.
            //gpu_time = magma_sync_wtime( opts.queue );
            cudaStreamSynchronize(cuda_stream[0]);
            cudaStreamSynchronize(cuda_stream[1]);
            cudaStreamSynchronize(cuda_stream[2]);

            //magma_sgesv_batched
            info = linearSolverSLU_batched(N, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount,cuda_stream[0]);
            cudaStreamSynchronize(cuda_stream[0]);

            //gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            //gpu_perf = gflops / gpu_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"

            //copy from gpu to host, cublasGetVectorAsync + cudaStreamSynchronize
            //magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1, opts.queue );

            cublasGetVectorAsync(
                int(batchCount), sizeof(int),
                dinfo_array, 1,
                cpu_info, 1, cuda_stream[0]);
            cudaStreamSynchronize(cuda_stream[0]);

            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_sgesv_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_sgesv_batched returned argument error %lld: %s.\n",
                        (long long) info, "magma_strerror( info )");
            }
            
            //=====================================================================
            // Residual
            //=====================================================================
            //copy result array B from gpu to array x on host.
            //cublasGetMatrixAsync + cudaStreamSynchronize
            cublasGetMatrixAsync(
                int(N), int(nrhs *batchCount), sizeof(float),
                d_B, int(lddb),
                h_X, int(ldb), cuda_stream[0]);
            cudaStreamSynchronize(cuda_stream[0]);

            double endTime = magma_wtime();
            double elapsedTime = endTime - startTime;
            printf("Time elapsed:%f", elapsedTime);
            //magma_sgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );

            /*error = 0;
            for (magma_int_t s=0; s < batchCount; s++)
            {
                Anorm = lapackf77_slange("I", &N, &N,    h_A + s * lda * N, &lda, work);
                Xnorm = lapackf77_slange("I", &N, &nrhs, h_X + s * ldb * nrhs, &ldb, work);
            
                blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A + s * lda * N, &lda,
                                       h_X + s * ldb * nrhs, &ldb,
                           &c_neg_one, h_B + s * ldb * nrhs, &ldb);
            
                Rnorm = lapackf77_slange("I", &N, &nrhs, h_B + s * ldb * nrhs, &ldb, work);
                float err = Rnorm/(N*Anorm*Xnorm);
                
                if (std::isnan(err) || std::isinf(err)) {
                    error = err;
                    break;
                }
                error = max( err, error );
            }
            bool okay = (error < tol);
            status += ! okay;**/

            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            /*if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_sgesv( &N, &nrhs, h_A + s * lda * N, &lda, ipiv + s * N, h_B + s * ldb * nrhs, &ldb, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_sgesv matrix %lld returned error %lld: %s.\n",
                                (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                printf( "%10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs,
                        gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }*/
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );
            magma_free_cpu( cpu_info );
            
            magma_free( d_A );
            magma_free( d_B );

            magma_free( dipiv );
            magma_free( dinfo_array );

            magma_free( dA_array );
            magma_free( dB_array );
            magma_free( dipiv_array );
            fflush( stdout );
        }
        /*if ( opts.niter > 1 ) {
            printf( "\n" );
        }*/
    }

    //opts.cleanup();
    magma_finalize();
    return status;
}
