#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "2DArray.h"
#define MM 1000
#define NN 1000
#define PP 1000

double **transpose(double **B, int pf, int pl, int nf, int nl)
{
    double **T = Allocate2DArray<double>((nl - nf), (pl - pf));
#pragma omp parallel for
    for (int i = 0; i < (nl - nf); i++)
    {
        for (int j = 0; j < (pl - pf); j++)
        {
            T[i][j] = B[j + pf][i + nf];
        }
    }
    return T;
}

void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double **A, double **B, double **C)
{
#pragma omp parallel for
    for (int i = mf; i < ml; i++)
        for (int j = nf; j < nl; j++)
        {
            // double * a = A[i];
            // double * b = B[j];
            double sum=0;
            C[i][j]=0;
            #pragma omp simd reduction(+:sum)
            for (int k = pf; k < pl; k++)
            {
                // sum += a[k] * b[k]; //using B transpose
                sum += A[i][k]*B[j][k];
            }
            C[i][j] += sum; 
        }
}

int main(int argc, char *argv[])
{
    int i, j,LOOP_COUNT=10;
    double start, time1, time2;

    int M = MM;
    int N = NN;
    int P = PP;

    if(argc >= 4){ 
	    M = atoi(argv[1]);
	    P = atoi(argv[2]);
	    N = atoi(argv[3]);
    }
    if(argc==5) LOOP_COUNT = atoi(argv[4]);
    double **A = Allocate2DArray<double>(M, P);
    // double **B = Allocate2DArray<double>(P, N);
    double **BT = Allocate2DArray<double>(N, P);

    double **C = Allocate2DArray<double>(M, N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < P; j++)
        {
            A[i][j] = (i * P + j + 1);
        }
    }

    // for (i = 0; i < P; i++)
    // {
    //     for (j = 0; j < N; j++)
    //     {
    //         B[i][j] = (-i * N - j - 1);
    //     }
    // }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j++)
        {
            BT[i][j] = (-i * P - j - 1);
        }
    }

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i][j] = 0.0 ;
        }
    }

    printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);

    printf("Execute matmult parallel\n");
    matmultleaf(0,M,0,N,0,P,A,BT,C);// using B transpose
    start = omp_get_wtime();
    for(i=0;i<LOOP_COUNT;i++){
        matmultleaf(0,M,0,N,0,P,A,BT,C);// using B transpose
    }
    time1 = (omp_get_wtime() - start)/LOOP_COUNT;
    printf("Time = %.5f milli seconds\n\n", time1*1000);

    Free2DArray<double>(A);
    // Free2DArray<double>(B);
    Free2DArray<double>(BT);
    Free2DArray<double>(C);

  if (time1 < 0.9 / LOOP_COUNT)
  {
    time1 = 1.0 / LOOP_COUNT / time1;
    i = (int)(time1* LOOP_COUNT) + 1;
    printf(" It is highly recommended to define LOOP_COUNT for this example on "
           "your \n"
           " computer as %i to have total execution time about 1 second for "
           "reliability \n"
           " of measurements\n\n",
           i);
  }

    return 0;
}
