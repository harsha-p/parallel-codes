#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "2DArray.h"
#define MM 1000
#define NN 1000
#define PP 1000

double dabs(double d) { return (d < 0.0 ? d : (-d)); }
int max(int a, int b) { return (a > b ? a : b); }
void matmult1(int m, int n, int p, double **A, double **B, double **C)
{
    int i, j, k;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < p; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

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
// mf, ml; /* first and last+1 i index */
// nf, nl; /* first and last+1 j index */
// pf, pl; /* first and last+1 k index */
{
    // double **BT = transpose(B, pf, pl, nf, nl);
#pragma omp parallel for collapse(2)
    for (int i = mf; i < ml; i++)
        for (int j = nf; j < nl; j++)
        {
            double sum;
#pragma omp reduce(+ \
                   : sum)
            for (int k = pf; k < pl; k++)
            {
                sum += A[i][j] * B[k][j];
                // sum += A[i][k] * BT[j - nf][k - pf];
            }
            C[i][j] += sum; // + or no ?
        }
}

int GRAIN = 32768; /* product size below which matmultleaf is used */
// 32768  <- pow 15
// 65536
// 131072
// 262144
// 524288

void matmultrec(int mf, int ml, int nf, int nl, int pf, int pl, double **A, double **B, double **C)
/*    
  recursive subroutine to compute the product of two  
  submatrices of A and B and store the result in C  
*/
// mf, ml; /* first and last+1 i index */
// nf, nl; /* first and last+1 j index */
// pf, pl; /* first and last+1 k index */

{
    //
    // Check sizes of matrices;
    // if below threshold then compute product w/o recursion
    //
    if ((ml - mf) * (nl - nf) * (pl - pf) < GRAIN)
        matmultleaf(mf, ml, nf, nl, pf, pl, A, B, C);
    else
    {
        if ((ml - mf) >= max((pl - pf), (nl - nf)))
        {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
            {
                // C0[01] = A0 * B
                matmultrec(mf, mf + (ml - mf) / 2, nf, nl, pf, pl, A, B, C);
                // C1[01] = A1 * B
                matmultrec(mf + (ml - mf) / 2, ml, nf, nl, pf, pl, A, B, C);
            }
        }
        else if ((pl - pf) >= max((nl - nf), (ml - mf)))
        {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
            {
                // C= A0*B0 + A1*B1
                matmultrec(mf, ml, nf, nl, pf, pf + (pl - pf) / 2, A, B, C);
                matmultrec(mf, ml, nf, nl, pf + (pl - pf) / 2, pl, A, B, C);
            }
        }
        else if ((nl - nf) >= max((nl - nf), (pl - pf)))
        {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
            {
                //C0 =A*B0
                matmultrec(mf, ml, nf, nf + (nl - nf) / 2, pf, pl, A, B, C);
                //C1 = A*B2
                matmultrec(mf, ml, nf + (nl - nf) / 2, nl, pf, pl, A, B, C);
            }
        }
#pragma omp taskwait
    }
}

int CheckResults(int m, int n, double **C, double **C1)
{
#define ERR_THRESHOLD 0.001
    int code = 0;
    //
    //  May need to take into consideration the floating point roundoff error
    //    due to parallel execution
    //
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dabs(C[i][j] - C1[i][j]) > ERR_THRESHOLD)
            {
                printf("%f  %f at [%d][%d]\n", C[i][j], C1[i][j], i, j);
                code = 1;
            }
        }
    }
    return code;
}

int main(int argc, char *argv[])
{
    int i, j;
    double start, time1, time2;

    int M = MM;
    int N = NN;
    int P = PP;

    if(argc >= 4){ 
	    M = atoi(argv[1]);
	    P = atoi(argv[2]);
	    N = atoi(argv[3]);
    }
    if(argc==5) GRAIN=atoi(argv[4]);
    
    double **A = Allocate2DArray<double>(M, P);
    double **B = Allocate2DArray<double>(P, N);

    double **C1 = Allocate2DArray<double>(M, N);
    double **C4 = Allocate2DArray<double>(M, N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < P; j++)
        {
            A[i][j] = (i * P + j + 1);
        }
    }

    for (i = 0; i < P; i++)
    {
        for (j = 0; j < N; j++)
        {
            B[i][j] = (-i * N - j);
        }
    }

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            C1[i][j] = (-i * N - j);
            C4[i][j] = (-i * N - j);
        }
    }

    printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
    // printf("Execute matmult1\n");
    // start = omp_get_wtime();
    // matmult1(M, N, P, A, B, C1);
    // time1 = omp_get_wtime() - start;
    // printf("Time = %f seconds\n\n", time1);

    printf("Execute matmultr\n");
    printf("%d GRAIN\n",GRAIN);
    start = omp_get_wtime();
    matmultrec(0, M, 0, N, 0, P, A, B, C4);
    time2 = omp_get_wtime() - start;
    printf("Time = %f seconds\n\n", time2);

    // printf("Checking...");
    // if (CheckResults(M, N, C1, C4))
    //     printf("Error in Recursive Matrix Multiplication\n\n");
    // else
    // {
    //     printf("OKAY\n\n");
    //     printf("Speedup = %5.1fX\n", time1 / time2);
    // }

    Free2DArray<double>(A);
    Free2DArray<double>(B);
    // Free2DArray<double>(C1);
    Free2DArray<double>(C4);

    return 0;
}
