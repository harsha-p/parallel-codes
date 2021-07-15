#include <bits/stdc++.h>
#include "omp.h"
using namespace std;

void mat_mul(double *A, double *B, double *C, int m, int n, int p)
{
    double sum;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum = 0.0;
#pragma omp reduce(+ : sum)
            for (int k = 0; k < p; k++)
                sum += A[p * i + k] * B[n * k + j];
            C[n * i + j] = sum;
        }
    }
}

int main(int argc, char const *argv[])
{
    // A m*p, B p*n, C m*n
    int m, n, p, i;
    m = 2000;
    p = 200;
    n = 1000;
    if(argc == 4){ 
	    m = atoi(argv[1]);
	    p = atoi(argv[2]);
	    n = atoi(argv[3]);
    }
    double *A, *B, *C;
    A = (double *)malloc(m * p * sizeof(double));
    B = (double *)malloc(p * n * sizeof(double));
    C = (double *)malloc(m * n * sizeof(double));
    if (A == NULL || B == NULL || C == NULL)
    {
        cout << "Error allocating memory for matrices\n";
        free(A);
        free(B);
        free(C);
        return 0;
    }
    for (i = 0; i < (m * p); i++)
    {
        A[i] = (double)(i + 1);
    }
    for (i = 0; i < (p * n); i++)
    {
        B[i] = (double)(-i - 1);
    }
    for (i = 0; i < (m * n); i++)
    {
        C[i] = 0.0;
    }
    double start_time = omp_get_wtime();
    mat_mul(A, B, C, m, n, p);
    double end_time = omp_get_wtime();
    printf("time taken %f\n", end_time - start_time);
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            printf("%12.5G", C[j + i * n]);
        }
        printf("\n");
    }
    return 0;
}
