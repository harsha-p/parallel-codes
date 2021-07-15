#include <stdlib.h>
#include <cstdio>
#include <omp.h>
#include "2DArray.h"
#define MM 1000
#define NN 1000
#define PP 1000

int min(int a, int b) { return a < b ? a : b; }

void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double *A, double *B, double *C, int M, int N, int P)
{
#pragma omp parallel for collapse(2)
    for (int i = mf; i < ml; i++)
        for (int j = nf; j < nl; j++)
        {
            double sum = 0;
#pragma omp reduction(+ \
                      : sum)
            for (int k = pf; k < pl; k++)
            {
                // sum += A[i * P + j] * B[j * P + k];
                sum += A[i * P + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
}

int main(int argc, char *argv[])
{
    int i, j, LOOP_COUNT = 10;
    double start, time1, time2;

    int M = MM;
    int N = NN;
    int P = PP;

    if (argc >= 4)
    {
        M = atoi(argv[1]);
        P = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    if (argc == 5)
        LOOP_COUNT = atoi(argv[4]);

    double *A = (double *)malloc(M * P * sizeof(double));
    double *B = (double *)malloc(P * N * sizeof(double));
    double *C = (double *)malloc(M * N * sizeof(double));

#pragma omp parallel for schedule(static)
    for (i = 0; i < M * P; i++)
        A[i] = (double)(i + 1);
#pragma omp parallel for schedule(static)
    for (i = 0; i < P * N; i++)
        B[i] = (double)(-i - 1);
#pragma omp parallel for schedule(static)
    for (i = 0; i < M * N; i++)
        C[i] = 0.0;
    printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);

    printf("Execute matmult parallel\n");
    // matmultleaf(0, M, 0, N, 0, P, A, B, C, M, N, P); // using B transpose
    start = omp_get_wtime();
    for (i = 0; i < LOOP_COUNT; i++)
    {
        matmultleaf(0, M, 0, N, 0, P, A, B, C, M, N, P); // using B transpose
    }
    time1 = (omp_get_wtime() - start) / LOOP_COUNT;
    printf("Time = %.5f milli seconds\n\n", time1 * 1000);

    for (int i = 0; i < min(6, M); i++)
    {
        for (int j = 0; j < min(6, P); j++)
        {
            printf("%12.5G", A[i * P + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < min(6, P); i++)
    {
        for (int j = 0; j < min(6, N); j++)
        {
            printf("%12.5G", B[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < min(6, M); i++)
    {
        for (int j = 0; j < min(6, N); j++)
        {
            printf("%12.5G", C[i * N + j]);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);

    if (time1 < 0.9 / LOOP_COUNT)
    {
        time1 = 1.0 / LOOP_COUNT / time1;
        i = (int)(time1 * LOOP_COUNT) + 1;
        printf(" It is highly recommended to define LOOP_COUNT for this example on "
               "your \n"
               " computer as %i to have total execution time about 1 second for "
               "reliability \n"
               " of measurements\n\n",
               i);
    }

    return 0;
}
