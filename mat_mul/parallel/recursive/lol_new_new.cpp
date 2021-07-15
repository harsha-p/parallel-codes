#include "2DArray.h"
#include "mkl.h"
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define MM 1000
#define NN 1000
#define PP 1000
int M, N, P;
int max(int a, int b) { return (a > b ? a : b); }
int cnt = 0;
void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double *A,
                 double *B, double *C) {
#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(A, B, C, mf, ml, nf, nl, pf, pl, P, N)
  for (int i = mf; i < ml; i++)
    for (int j = nf; j < nl; j++) {
      double sum = 0;
#pragma omp simd reduction(+ : sum)
      for (int k = pf; k < pl; k++) {
        sum += A[i * P + k] * B[j * P + k];
      }
      C[i * N + j] = sum;
    }
}

double LIMIT = 2097152; /* product size below which matmultleaf is used */
// 32768  <- pow 15
// 65536
// 131072
// 262144
// 524288

void matmultrec(int mf, int ml, int nf, int nl, int pf, int pl, double *A,
                double *B, double *C) {
  double mul = (double)(ml - mf);
  mul *= (nl - nf);
  mul *= (pl - pf);
  if (mul < LIMIT)
    matmultleaf(mf, ml, nf, nl, pf, pl, A, B, C);
  else {
    if ((ml - mf) >= max((pl - pf), (nl - nf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        cnt++;
        // C0[01] = A0 * B
        matmultrec(mf, mf + (ml - mf) / 2, nf, nl, pf, pl, A, B, C);
      }
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        // C1[01] = A1 * B
        matmultrec(mf + (ml - mf) / 2, ml, nf, nl, pf, pl, A, B, C);
      }
    } else if ((pl - pf) >= max((nl - nf), (ml - mf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        cnt++;
        // C= A0*B0 + A1*B1
      }
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        matmultrec(mf, ml, nf, nl, pf, pf + (pl - pf) / 2, A, B, C);
        matmultrec(mf, ml, nf, nl, pf + (pl - pf) / 2, pl, A, B, C);
      }
    } else if ((nl - nf) >= max((nl - nf), (pl - pf))) {
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        cnt++;
        // C0 =A*B0
        matmultrec(mf, ml, nf, nf + (nl - nf) / 2, pf, pl, A, B, C);
      }
#pragma omp task firstprivate(mf, ml, nf, nl, pf, pl)
      {
        // C1 = A*B2
        matmultrec(mf, ml, nf + (nl - nf) / 2, nl, pf, pl, A, B, C);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int i, j, LOOP_COUNT = 10;
  double start, time1, time2;

  M = MM;
  N = NN;
  P = PP;

  if (argc >= 4) {
    M = atoi(argv[1]);
    P = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  if (argc >= 5)
    LIMIT = atof(argv[4]);
  if (argc == 6)
    LOOP_COUNT = atoi(argv[5]);

  /* double *A = (double *)MKL_malloc(M * P * sizeof(double), 64); */
  /* double *BT = (double *)MKL_malloc(N * P * sizeof(double), 64); */
  /* double *C = (double *)MKL_malloc(M * N * sizeof(double), 64); */
  double *A, *BT, *C;
  posix_memalign((void **)&A, 64, M * P * sizeof(double));
  posix_memalign((void **)&BT, 64, N * P * sizeof(double));
  posix_memalign((void **)&C, 64, M * N * sizeof(double));
  // double * A =(double *)malloc(M*P*sizeof(double));
  // double * BT =(double *)malloc(N*P*sizeof(double));
  // double * C =(double *)malloc(M*N*sizeof(double));

#pragma omp parallel
  {
#pragma omp for
    for (i = 0; i < M * P; i++) {
      A[i] = (double)(i + 1);
    }

#pragma omp for
    for (i = 0; i < N * P; i++) {
      BT[i] = (double)(-j - 1);
    }

#pragma omp for
    for (i = 0; i < M * N; i++) {
      C[i] = 0.0;
    }
  }

  printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
  printf("%f LIMIT\n", LIMIT);

  printf("Execute matmult recursive\n");
  start = omp_get_wtime();
  for (i = 0; i < LOOP_COUNT; i++) {
    matmultrec(0, M, 0, N, 0, P, A, BT, C);
  }
  time1 = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("%d cnt\n", cnt);
  printf("%d cnt per run\n", cnt / LOOP_COUNT);
  printf("Time = %.5f milli seconds\n\n", time1 * 1000);

  /* mkl_free(A); */
  /* mkl_free(BT); */
  /* mkl_free(C); */
  free(A);
  free(BT);
  free(C);
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
