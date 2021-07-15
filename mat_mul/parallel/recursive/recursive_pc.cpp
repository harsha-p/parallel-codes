#include "2DArray.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define MM 1000
#define NN 1000
#define PP 1000

int max(int a, int b) { return (a > b ? a : b); }

void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double **A,
                 double **B, double **C) {
  double *a, *b;
  int size_m = (ml - mf);
  int size_p = (pl - pf);
  int size_n = (nl - nf);
  a = (double *)malloc(size_m * size_p * sizeof(double));
  b = (double *)malloc(size_p * size_n * sizeof(double));
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = mf; i < ml; i++) {
      for (int j = pf; j < pl; j++) {
        a[(i - mf) * (size_p) + j - pf] = A[i][j];
      }
    }
#pragma omp for schedule(dynamic)
    for (int i = nf; i < nl; i++) {
      for (int j = pf; j < pl; j++) {
        b[(i - nf) * (size_p) + j - pf] = B[i][j];
      }
    }
  }
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < size_m; i++)
    for (int j = 0; j < size_n; j++) {
      double sum = 0;
#pragma omp simd reduction(+ : sum)
      for (int k = 0; k < size_p; k++) {
        sum += a[i * size_p + k] * b[j * size_p + k];
      }
      C[i + mf][j + nf] += sum;
    }
}

int LIMIT = 32768; /* product size below which matmultleaf is used */
// 32768  <- pow 15
// 65536
// 131072
// 262144
// 524288

void matmultrec(int mf, int ml, int nf, int nl, int pf, int pl, double **A,
                double **B, double **C) {
  if ((ml - mf) * (nl - nf) * (pl - pf) < LIMIT)
    matmultleaf(mf, ml, nf, nl, pf, pl, A, B, C);
  else {
    if ((ml - mf) >= max((pl - pf), (nl - nf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        // C0[01] = A0 * B
        matmultrec(mf, mf + (ml - mf) / 2, nf, nl, pf, pl, A, B, C);
        // C1[01] = A1 * B
        matmultrec(mf + (ml - mf) / 2, ml, nf, nl, pf, pl, A, B, C);
      }
    } else if ((pl - pf) >= max((nl - nf), (ml - mf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        // C= A0*B0 + A1*B1
        matmultrec(mf, ml, nf, nl, pf, pf + (pl - pf) / 2, A, B, C);
        matmultrec(mf, ml, nf, nl, pf + (pl - pf) / 2, pl, A, B, C);
      }
    } else if ((nl - nf) >= max((nl - nf), (pl - pf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        // C0 =A*B0
        matmultrec(mf, ml, nf, nf + (nl - nf) / 2, pf, pl, A, B, C);
        // C1 = A*B2
        matmultrec(mf, ml, nf + (nl - nf) / 2, nl, pf, pl, A, B, C);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int i, j, LOOP_COUNT = 10;
  double start, time1, time2;

  int M = MM;
  int N = NN;
  int P = PP;

  if (argc >= 4) {
    M = atoi(argv[1]);
    P = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  if (argc == 5)
    LIMIT = atoi(argv[4]);
  if (argc == 6)
    LOOP_COUNT = atoi(argv[5]);

  double **A = Allocate2DArray<double>(M, P);
  // double **B = Allocate2DArray<double>(P, N);
  double **BT = Allocate2DArray<double>(N, P);

  double **C = Allocate2DArray<double>(M, N);

#pragma omp parallel
  {
    printf("%d num\n", omp_get_num_threads());
#pragma omp for
    for (i = 0; i < M; i++) {
      for (j = 0; j < P; j++) {
        A[i][j] = (i * P + j + 1);
      }
    }

#pragma omp for
    for (i = 0; i < N; i++) {
      for (j = 0; j < P; j++) {
        BT[i][j] = (-i * P - j - 1);
      }
    }

#pragma omp for
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        C[i][j] = 0.0;
      }
    }
  }

  printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
  printf("%d LIMIT\n", LIMIT);

  printf("Execute matmult recursive\n");
  start = omp_get_wtime();
  for (i = 0; i < LOOP_COUNT; i++) {
    matmultrec(0, M, 0, N, 0, P, A, BT, C);
  }
  time1 = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("Time = %.5f milli seconds\n\n", time1 * 1000);

  Free2DArray<double>(A);
  Free2DArray<double>(BT);
  Free2DArray<double>(C);

  if (time1 < 0.9 / LOOP_COUNT) {
    time1 = 1.0 / LOOP_COUNT / time1;
    i = (int)(time1 * LOOP_COUNT) + 1;
    printf(" It is highly recommended to define LOOP_COUNT for this example "
           "on "
           "your \n"
           " computer as %i to have total execution time about 1 second for "
           "reliability \n"
           " of measurements\n\n",
           i);
  }
  return 0;
}
