#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int M, N, P;

#define min(a, b) (((a) < (b) ? (a) : (b)))
#define max(a, b) (((a) > (b) ? (a) : (b)))

#define tile1 128
#define tile2 64
#define tile3 16

void mat_mult_block(int mf, int ml, int nf, int nl, int pf, int pl,
                    double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C) {
  for (int i = mf; i < ml; i++) {
    for (int j = nf; j < nl; j++) {
      for (int k = pf; k < pl; k++) {
        C[i * N + j] += A[i * P + k] * B[j * P + k];
      }
    }
  }
}

void tiled_level_3(int im, int jm, int km, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int il, jl, kl;
  for (il = 0; il < tile2; il += tile3) {
    for (jl = 0; jl < tile2; jl += tile3) {
      for (kl = 0; kl < tile2; kl += tile3) {
        mat_mult_block(im + il, im + il + tile2, jm + jl, jm + jl + tile2,
                       km + kl, km + kl + tile2, A, B, C);
      }
    }
  }
}

void tiled_level_2(int ih, int jh, int kh, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int im, jm, km;
  for (im = 0; im < tile1; im += tile2) {
    for (jm = 0; jm < tile1; jm += tile2) {
      for (km = 0; km < tile1; km += tile2) {
        tiled_level_3(im + ih, jm + jh, km + kh, A, B, C);
        /* matmultleaf(ih+im , ih+im+tile2 , jh+jm , jh+jm+tile2 , kh+km ,
         * kh+km+tile2 , A, B, C); */
      }
    }
  }
}

void tiled_level_1(int mf, int ml, int nf, int nl, int pf, int pl,
                   double *__restrict__ A, double *__restrict__ B,
                   double *__restrict__ C) {
  int ih, jh, kh;
  for (ih = mf; ih < ml; ih += tile1) {
    for (jh = nf; jh < nl; jh += tile1) {
      for (kh = nf; kh < nl; kh += tile1) {
        tiled_level_2(ih, jh, kh, A, B, C);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int i, j, LOOP_COUNT = 10;
  double start, time;

  M = 1024;
  N = 1024;
  P = 1024;

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
  double *A = (double *)_mm_malloc(size_a, 32);
  double *BT = (double *)_mm_malloc(size_b, 32);
  double *C = (double *)_mm_malloc(size_c, 32);

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
    tiled_level_1(0, M, 0, N, 0, P, A, BT, C);
  }
  time = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("Time = %.5f milli seconds\n\n", time * 1000);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      printf("%12.5G", A[i * P + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      printf("%12.5G", BT[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
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
