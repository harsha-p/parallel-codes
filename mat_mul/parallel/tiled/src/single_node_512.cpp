#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int TILE1 = 64;
int TILE2 = 64;
int TILE3 = 64;
int TILING_LEVEL = 3;
int M, N, P;

void mat_mult_block(int tile3, double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ C) {
  int tN = 2 * N;
  int thN = 3 * N;
  int iloop = tile3 >> 2;
  int jloop = tile3 >> 4;
  for (int i = 0; i < iloop; i++) {
    int I = i << 2;
    int i_n = (i << 2) * N;
    for (int j = 0; j < jloop; j++) {
      int J = j << 4;
      __m512d c0_0 = _mm512_load_pd(C + i_n + J);
      __m512d c0_1 = _mm512_load_pd(C + i_n + J + 8);
      __m512d c1_0 = _mm512_load_pd(C + i_n + N + J);
      __m512d c1_1 = _mm512_load_pd(C + i_n + N + J + 8);
      __m512d c2_0 = _mm512_load_pd(C + i_n + tN + J);
      __m512d c2_1 = _mm512_load_pd(C + i_n + tN + J + 8);
      __m512d c3_0 = _mm512_load_pd(C + i_n + thN + J);
      __m512d c3_1 = _mm512_load_pd(C + i_n + thN + J + 8);
      for (int k = 0; k < tile3; k++) {
        int K = k * P;
        __m512d b0 = _mm512_load_pd(B + K + J);
        __m512d b1 = _mm512_load_pd(B + K + J + 8);
        __m512d b2 = _mm512_load_pd(B + K + J + 16);
        __m512d b3 = _mm512_load_pd(B + K + J + 24);
        __m512d a00 = _mm512_set1_pd(*(A + i_n + k));
        c0_0 = _mm512_fmadd_pd(a00, b0, c0_0);
        c0_1 = _mm512_fmadd_pd(a00, b1, c0_1);
        a00 = _mm512_set1_pd(*(A + i_n + N + k));
        c1_0 = _mm512_fmadd_pd(a00, b0, c1_0);
        c1_1 = _mm512_fmadd_pd(a00, b1, c1_1);
        a00 = _mm512_set1_pd(*(A + i_n + tN + k));
        c2_0 = _mm512_fmadd_pd(a00, b0, c2_0);
        c2_1 = _mm512_fmadd_pd(a00, b1, c2_1);
        a00 = _mm512_set1_pd(*(A + i_n + thN + k));
        c3_0 = _mm512_fmadd_pd(a00, b0, c3_0);
        c3_1 = _mm512_fmadd_pd(a00, b1, c3_1);
      }
      _mm512_store_pd(C + i_n + J, c0_0);
      _mm512_store_pd(C + i_n + J + 8, c0_1);
      _mm512_store_pd(C + i_n + N + J, c1_0);
      _mm512_store_pd(C + i_n + N + J + 8, c1_1);
      _mm512_store_pd(C + i_n + tN + J, c2_0);
      _mm512_store_pd(C + i_n + tN + J + 8, c2_1);
      _mm512_store_pd(C + i_n + thN + J, c3_0);
      _mm512_store_pd(C + i_n + thN + J + 8, c3_1);
    }
  }
}

void tiled_level_3(int tile2, int tile3, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int t3t2 = tile2 / tile3;
  for (int ih = 0; ih < t3t2; ih++) {
    for (int jh = 0; jh < t3t2; jh++) {
      for (int kh = 0; kh < t3t2; kh++) {
        mat_mult_block(TILE3, A + ih * tile3 * P + jh * tile3,
                       B + jh * tile3 + kh * tile3 * P,
                       C + ih * tile3 + kh * tile3 * P);
      }
    }
  }
}

void tiled_level_2(int tile1, int tile2, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int t2t1 = tile1 / tile2;
  for (int im = 0; im < t2t1; im++) {
    for (int jm = 0; jm < t2t1; jm++) {
      for (int km = 0; km < t2t1; km++) {
        tiled_level_3(tile2, TILE3, A + im * tile2 * P + jm * tile2,
                      B + jm * tile2 + km * tile2 * P,
                      C + im * tile2 + km * tile2 * P);
      }
    }
  }
}

void tiled_level_1(int tile1, int m, int n, int p, double *__restrict__ A,
                   double *__restrict__ B, double *__restrict__ C) {
  int im = m / tile1, jn = n / tile1, kp = p / tile1;
  for (int i = 0; i < im; i++) {
    for (int j = 0; j < jn; j++) {
      for (int k = 0; k < kp; k++) {
        tiled_level_2(tile1, TILE2, A + i * tile1 * P + j * tile1,
                      B + j * tile1 + k * tile1 * P,
                      C + i * tile1 + k * tile1 * P);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int i, j, LOOP_COUNT = 100;
  double start, time;

  M = 1024;
  N = 1024;
  P = 1024;
  // matrix size ; loop count; tiling level; tile sizes;

  if (argc < 7) {
    printf("not enough arguments\n");
    return 0;
  }
  M = N = P = atoi(argv[1]);
  LOOP_COUNT = atoi(argv[2]);
  TILING_LEVEL = atoi(argv[3]);
  TILE1 = atoi(argv[4]);
  TILE2 = atoi(argv[5]);
  TILE3 = atoi(argv[6]);

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

  printf("Using AVX 512\n");
  printf("Matrix: M = %d  P = %d  N = %d\n", M, P, N);
  printf("LOOP_COUNT: %d TILING_LEVEL: %d\n", LOOP_COUNT, TILING_LEVEL);
  printf("TILE SIZES: TILE1 = %d  TILE2 = %d  TILE3 = %d\n", TILE1, TILE2,
         TILE3);

  if (TILING_LEVEL == 1) {
    start = omp_get_wtime();
    for (i = 0; i < LOOP_COUNT; i++) {
      tiled_level_3(M, TILE3, A, B, C);
    }
    time = (omp_get_wtime() - start) / LOOP_COUNT;
  } else if (TILING_LEVEL == 2) {
    start = omp_get_wtime();
    for (i = 0; i < LOOP_COUNT; i++) {
      tiled_level_2(M, TILE2, A, B, C);
    }
    time = (omp_get_wtime() - start) / LOOP_COUNT;
  } else if (TILING_LEVEL == 3) {
    start = omp_get_wtime();
    for (i = 0; i < LOOP_COUNT; i++) {
      tiled_level_1(TILE1, M, N, P, A, B, C);
    }
    time = (omp_get_wtime() - start) / LOOP_COUNT;
  } else if (TILING_LEVEL == 0) {
    start = omp_get_wtime();
    for (i = 0; i < LOOP_COUNT; i++) {
      mat_mult_block(M, A, B, C);
    }
    time = (omp_get_wtime() - start) / LOOP_COUNT;
  } else {
    printf("invalid tiling level\n");
    return 0;
  }

  printf("TIME: %.5f\n\n", time * 1000);
  fprintf(stderr, "%.5f\n", time * 1000);

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
    fprintf(stderr, "%i\n", i);
  }
  return time;
}
