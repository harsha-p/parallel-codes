#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b) ? (a) : (b)))
#define max(a, b) (((a) > (b) ? (a) : (b)))

int TILE1 =  64;
int TILE2 =  64;
int TILE3 =  16;
//145 for l3 t3=32 , t3016
//150 for l2 t3=32 t2=64
int M, N, P;

void mat_mult_block(int tile3, double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C) {
  for (int i = 0; i < tile3 / 4; i++) {
    for (int j = 0; j < tile3 / 16; j++) {
      __m256d c0_0 = _mm256_load_pd(C + ((4 * i + 0) * N) + j * 16 + 0);
      __m256d c0_1 = _mm256_load_pd(C + ((4 * i + 0) * N) + j * 16 + 1);
      __m256d c0_2 = _mm256_load_pd(C + ((4 * i + 0) * N) + j * 16 + 2);
      __m256d c0_3 = _mm256_load_pd(C + ((4 * i + 0) * N) + j * 16 + 3);
      __m256d c1_0 = _mm256_load_pd(C + ((4 * i + 1) * N) + j * 16 + 0);
      __m256d c1_1 = _mm256_load_pd(C + ((4 * i + 1) * N) + j * 16 + 1);
      __m256d c1_2 = _mm256_load_pd(C + ((4 * i + 1) * N) + j * 16 + 2);
      __m256d c1_3 = _mm256_load_pd(C + ((4 * i + 1) * N) + j * 16 + 3);
      __m256d c2_0 = _mm256_load_pd(C + ((4 * i + 2) * N) + j * 16 + 0);
      __m256d c2_1 = _mm256_load_pd(C + ((4 * i + 2) * N) + j * 16 + 1);
      __m256d c2_2 = _mm256_load_pd(C + ((4 * i + 2) * N) + j * 16 + 2);
      __m256d c2_3 = _mm256_load_pd(C + ((4 * i + 2) * N) + j * 16 + 3);
      __m256d c3_0 = _mm256_load_pd(C + ((4 * i + 3) * N) + j * 16 + 0);
      __m256d c3_1 = _mm256_load_pd(C + ((4 * i + 3) * N) + j * 16 + 1);
      __m256d c3_2 = _mm256_load_pd(C + ((4 * i + 3) * N) + j * 16 + 2);
      __m256d c3_3 = _mm256_load_pd(C + ((4 * i + 3) * N) + j * 16 + 3);
      for (int k = 0; k < tile3; k++) {
        /* __m256d b0 = _mm256_loadu_pd(B + (k * P) + j * 16); */
        /* __m256d b1 = _mm256_loadu_pd(B + (k * P) + j * 16 + 4); */
        /* __m256d b2 = _mm256_loadu_pd(B + (k * P) + j * 16 + 8); */
        /* __m256d b3 = _mm256_loadu_pd(B + (k * P) + j * 16 + 12); */
        __m256d b0 = _mm256_load_pd(B + k + P*(j * 16));
        __m256d b1 = _mm256_load_pd(B + k + P*(j * 16 + 4));
        __m256d b2 = _mm256_load_pd(B + k + P*(j * 16 + 8));
        __m256d b3 = _mm256_load_pd(B + k + P*(j * 16 + 12));
        __m256d a00 = _mm256_broadcast_sd(A + ((4 * i) * N) + k);
        c0_0 = _mm256_fmadd_pd(a00, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a00, b0, c0_1);
        c0_2 = _mm256_fmadd_pd(a00, b0, c0_2);
        c0_3 = _mm256_fmadd_pd(a00, b0, c0_3);
        a00 = _mm256_broadcast_sd(A + ((4 * i + 1) * N) + k);
        c1_0 = _mm256_fmadd_pd(a00, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a00, b0, c1_1);
        c1_2 = _mm256_fmadd_pd(a00, b0, c1_2);
        c1_3 = _mm256_fmadd_pd(a00, b0, c1_3);
        a00 = _mm256_broadcast_sd(A + ((4 * i + 2) * N) + k);
        c2_0 = _mm256_fmadd_pd(a00, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a00, b0, c2_1);
        c2_2 = _mm256_fmadd_pd(a00, b0, c2_2);
        c2_3 = _mm256_fmadd_pd(a00, b0, c2_3);
        a00 = _mm256_broadcast_sd(A + ((4 * i + 3) * N) + k);
        c3_0 = _mm256_fmadd_pd(a00, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a00, b0, c3_1);
        c3_2 = _mm256_fmadd_pd(a00, b0, c3_2);
        c3_3 = _mm256_fmadd_pd(a00, b0, c3_3);
      }
      _mm256_store_pd(C + ((4 * i + 0) * N) + j * 16 + 0, c0_0);
      _mm256_store_pd(C + ((4 * i + 0) * N) + j * 16 + 1, c0_1);
      _mm256_store_pd(C + ((4 * i + 0) * N) + j * 16 + 2, c0_2);
      _mm256_store_pd(C + ((4 * i + 0) * N) + j * 16 + 3, c0_3);
      _mm256_store_pd(C + ((4 * i + 1) * N) + j * 16 + 0, c1_0);
      _mm256_store_pd(C + ((4 * i + 1) * N) + j * 16 + 1, c1_1);
      _mm256_store_pd(C + ((4 * i + 1) * N) + j * 16 + 2, c1_2);
      _mm256_store_pd(C + ((4 * i + 1) * N) + j * 16 + 3, c1_3);
      _mm256_store_pd(C + ((4 * i + 2) * N) + j * 16 + 0, c2_0);
      _mm256_store_pd(C + ((4 * i + 2) * N) + j * 16 + 1, c2_1);
      _mm256_store_pd(C + ((4 * i + 2) * N) + j * 16 + 2, c2_2);
      _mm256_store_pd(C + ((4 * i + 2) * N) + j * 16 + 3, c2_3);
      _mm256_store_pd(C + ((4 * i + 3) * N) + j * 16 + 0, c3_0);
      _mm256_store_pd(C + ((4 * i + 3) * N) + j * 16 + 1, c3_1);
      _mm256_store_pd(C + ((4 * i + 3) * N) + j * 16 + 2, c3_2);
      _mm256_store_pd(C + ((4 * i + 3) * N) + j * 16 + 3, c3_3);
    }
  }
}

void tiled_level_3(int tile2, int tile3, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int il, jl, kl;
  for (il = 0; il < tile2; il += tile3) {
    for (jl = 0; jl < tile2; jl += tile3) {
      for (kl = 0; kl < tile2; kl += tile3) {
        mat_mult_block(TILE3, A + il * P + jl, B + jl * P + kl,
                       C + il * P + kl);
      }
    }
  }
}

void tiled_level_2(int tile1, int tile2, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int im, jm, km;
  for (im = 0; im < tile1; im += tile2) {
    for (jm = 0; jm < tile1; jm += tile2) {
      for (km = 0; km < tile1; km += tile2) {
        tiled_level_3(tile2, TILE3, A + im * P + jm, B + jm * P + km,
                      C + im * P + km);
      }
    }
  }
}

void tiled_level_1(int tile1, int m, int n, int p, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int i, j, k;
  for (i = 0; i < m; i += tile1) {
    for (j = 0; j < n; j += tile1) {
      for (k = 0; k < p; k += tile1) {
        tiled_level_2(tile1, TILE2, A + i * P + j, B + j * P + k,
                      C + i * P + k);
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
    TILE1 = atoi(argv[1]);
    TILE2 = atoi(argv[2]);
    TILE3 = atoi(argv[3]);
  }
  if (argc == 5)
    LOOP_COUNT = atoi(argv[4]);
  if (argc == 2)
    LOOP_COUNT = atoi(argv[1]);

  int size_a = M * P * sizeof(double);
  int size_b = P * N * sizeof(double);
  int size_c = M * N * sizeof(double);
  double *A = (double *)_mm_malloc(size_a, 64);
  double *B = (double *)_mm_malloc(size_b, 64);
  double *C = (double *)_mm_malloc(size_c, 64);

  for (i = 0; i < M * P; i++) {
    A[i] = (double)(i + 1);
  }

  for (i = 0; i < N * P; i++) {
    B[i] = (double)(-i - 1);
  }

  for (i = 0; i < M * N; i++) {
    C[i] = 0.0;
  }

  printf("Matrix Dimensions: M = %d  P = %d  N = %d\n\n", M, P, N);
  start = omp_get_wtime();
  for (i = 0; i < LOOP_COUNT; i++) {
    tiled_level_1(TILE1, M, N, P, A, B, C);
    /* tiled_level_2(1024 , TILE2 , A, B, C); */
    /* tiled_level_3(1024, TILE3, A, B, C); */
    /* mat_mult_block(1024, A, B, C); */
  }
  time = (omp_get_wtime() - start) / LOOP_COUNT;
  printf("Time = %.5f milli seconds\n\n", time * 1000);
  printf("\n");
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      printf("%12.5G", C[i * N + j]);
    }
    printf("\n");
  }
  _mm_free(A);
  _mm_free(B);
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
