#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

int M, N, P;
int max(int a, int b) { return (a > b ? a : b); }

void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl, double * __restrict__ A,
                 double * __restrict__ B, double * __restrict__ C) {
  for (int i = mf; i < ml; i++){
	  int i_a = i*P;
    for (int j = nf; j < nl; j++) {
		int j_b = j*P;
      double sum = 0;
#pragma omp simd reduction(+:sum)
      for (int k = pf; k < pl; k++) {
        sum += A[i_a+k] * B[j_b + k];
      }
      C[i * N + j] += sum;
    }
  }
}

void tiled(int mf, int ml, int nf, int nl, int pf, int pl, double * __restrict__ A, double * __restrict__ B, double* __restrict__ C)
{
	int ih,jh,kh,im,jm,km,il,jl,kl;
	int tile1 =64, tile2=16;
#pragma omp parallel for collapse(2)
	for (ih=mf; ih < ml ; ih+=tile1) {
		for (jh = nf ; jh < nl ; jh +=tile1 ) {
			for (kh = nf ; kh < nl ; kh +=tile1 ) {
				for (im = 0; im < tile1 ; im+=tile2) {
					for (jm = 0 ; jm < tile1 ; jm +=tile2 ) {
						for (km = 0 ; km < tile1 ; km +=tile2 ) {
							// leaf multiplication
							// need to modify loops for irregular sizes of
							// matrices
							/* printf("%d %d %d %d %d %d\n",ih,jh,kh,im,jm,km); */

							matmultleaf(ih+im , ih+im+tile2 , jh+jm , jh+jm+tile2 , kh+km , kh+km+tile2 , A, B, C);
						}
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[]) {
  int i, j, LOOP_COUNT = 10;
  double start, time;

  M = 1000;
  N = 1000;
  P = 1000;

  if (argc >= 4) {
    M = atoi(argv[1]);
    P = atoi(argv[2]);
    N = atoi(argv[3]);
  }
  if (argc == 5)
    LOOP_COUNT = atoi(argv[4]);

  int size_a = M * P * sizeof(double);
  int size_b = P * N * sizeof(double);
  int size_c = M * N * sizeof(double);
  double *A = (double *)_mm_malloc(size_a, 64);
  double *BT = (double *)_mm_malloc(size_b, 64);
  double *C = (double *)_mm_malloc(size_c, 64);

  for (i = 0; i < M * P; i++) {
    A[i] = (double)(i + 1);
  }

  for (i = 0; i < N * P; i++) {
    BT[i] = (double)(-i - 1);
  }

  for (i = 0; i < M * N; i++) {
    C[i] = 0.0;
  }

  printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
  start = omp_get_wtime();
  for (i = 0; i < LOOP_COUNT; i++) {
    tiled(0, M, 0, N, 0, P, A, BT, C);
  }
  time = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("Time = %.5f milli seconds\n\n", time * 1000);
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j <  6 ; j++)
    {
      printf("%12.5G", A[i * P + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i <  6; i++)
  {
    for (int j = 0; j <6; j++)
    {
      printf("%12.5G", BT[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < 6; i++)
  {
    for (int j = 0; j <6; j++)
    {
      printf("%12.5G", C[i * N + j]);
    }
    printf("\n");
  }
  _mm_free(A);
  _mm_free(BT);
  _mm_free(C);

  if (time < 0.9 / LOOP_COUNT) {
    time = 1.0 / LOOP_COUNT / time;
    i = (int)(time * LOOP_COUNT) + 1;
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
