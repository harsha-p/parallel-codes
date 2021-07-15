#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#define MM 1000
#define NN 1000
#define PP 1000
int M, N, P;
int max(int a, int b) { return (a > b ? a : b); }
int cnt = 0;
void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double *restrict A, double *restrict B, double *restrict C)
{
  __assume_aligned(A, 64);
  __assume_aligned(B, 64);
  __assume_aligned(C, 64);
  // printf("%d %d %d %d %d %d\n", mf, ml, nf, nl ,pf,pl);
#pragma omp parallel for default(none) \
    shared(A, B, C, mf, ml, nf, nl, pf, pl, P, N) proc_bind(close)
  for (int i = mf; i < ml; i++)
    for (int j = nf; j < nl; j++)
    {
      double sum = 0;
#pragma omp simd aligned(A, B : 64) simdlen(16) reduction(+ \
                                                          : sum)
      for (int k = pf; k < pl; k++)
      {
        sum += A[i * P + k] * B[j * P + k];
      }
      C[i * N + j] = sum;
    }
}

double LIMIT = 2097152;

void matmultrec(int mf, int ml, int nf, int nl, int pf, int pl, double *A, double *B, double *C)
{
  double mul = (double)(ml - mf);
  mul *= (nl - nf);
  mul *= (pl - pf);
  if (mul < LIMIT)
  {
    /* matmultleaf(mf, ml, nf, nl, pf, pl, A, B, C); */
#pragma omp task default(none) firstprivate(mf, ml, nf, nl, pf, pl) shared(A, B, C, cnt)
    {
      cnt += 1;
      matmultleaf(mf, ml, nf, nl, pf, pl, A, B, C);
    }
  }
  else
  {
    if ((ml - mf) >= max((pl - pf), (nl - nf)))
    {
#pragma omp task default(none) firstprivate(mf, ml, nf, nl, pf, pl) shared(A, B, C, cnt)
      {
        cnt++;
        // C0[01] = A0 * B
        matmultrec(mf, mf + (ml - mf) / 2, nf, nl, pf, pl, A, B, C);
        // C1[01] = A1 * B
        matmultrec(mf + (ml - mf) / 2, ml, nf, nl, pf, pl, A, B, C);
      }
    }
    else if ((pl - pf) >= max((nl - nf), (ml - mf)))
    {
#pragma omp task default(none) firstprivate(mf, ml, nf, nl, pf, pl) shared(A, B, C, cnt)
      {
        cnt++;
        // C= A0*B0 + A1*B1
        matmultrec(mf, ml, nf, nl, pf, pf + (pl - pf) / 2, A, B, C);
        matmultrec(mf, ml, nf, nl, pf + (pl - pf) / 2, pl, A, B, C);
      }
    }
    else if ((nl - nf) >= max((nl - nf), (pl - pf)))
    {
#pragma omp task default(none) firstprivate(mf, ml, nf, nl, pf, pl) shared(A, B, C, cnt)
      {
        cnt++;
        // C0 =A*B0
        matmultrec(mf, ml, nf, nf + (nl - nf) / 2, pf, pl, A, B, C);
        // C1 = A*B2
        matmultrec(mf, ml, nf + (nl - nf) / 2, nl, pf, pl, A, B, C);
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int i, j, LOOP_COUNT = 10;
  double start, time1, time2;

  M = MM;
  N = NN;
  P = PP;

  if (argc >= 4)
  {
    M = atoi(argv[1]);
    P = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  if (argc >= 5)
    LIMIT = atof(argv[4]);
  if (argc == 6)
    LOOP_COUNT = atoi(argv[5]);

  int size_a = M * P * sizeof(double);
  int size_b = P * N * sizeof(double);
  int size_c = M * N * sizeof(double);
  // double *A = (double *)malloc(size_a);
  // double *BT = (double *)malloc(size_b);
  // double *C = (double *)malloc(size_c);
  double *A = (double *)_mm_malloc(size_a, 64);
  double *BT = (double *)_mm_malloc(size_b, 64);
  double *C = (double *)_mm_malloc(size_c, 64);
  /* double *A, *BT, *C; */
  /* posix_memalign((void **)&A, 64, size_a); */
  /* posix_memalign((void **)&BT, 64, size_b); */
  /* posix_memalign((void **)&C, 64, size_c); */

#pragma omp parallel
  {
#pragma omp for
    for (i = 0; i < M * P; i++)
    {
      A[i] = (double)(i + 1);
    }

#pragma omp for
    for (i = 0; i < N * P; i++)
    {
      BT[i] = (double)(-j - 1);
    }

#pragma omp for
    for (i = 0; i < M * N; i++)
    {
      C[i] = 0.0;
    }
  }

  printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
  printf("%f LIMIT\n", LIMIT);

  printf("Execute matmult recursive\n");
  start = omp_get_wtime();
  for (i = 0; i < LOOP_COUNT; i++)
  {
    matmultrec(0, M, 0, N, 0, P, A, BT, C);
  }
  time1 = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("%d cnt\n", cnt);
  printf("%d cnt per run\n", cnt / LOOP_COUNT);
  printf("Time = %.5f milli seconds\n\n", time1 * 1000);

  /* free(A); */
  /* free(BT); */
  /* free(C); */
  _mm_free(A);
  _mm_free(BT);
  _mm_free(C);

  if (time1 < 0.9 / LOOP_COUNT)
  {
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
