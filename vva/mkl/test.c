#define min(x, y) (((x) < (y)) ? (x) : (y))

#include "mkl.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  double *A, *B, *C;
  int n, i, j;
  double s_initial, s_elapsed;
  int LOOP_COUNT = 1;
  n = 1000;
  if (argc >= 2)
  {
    n = atoi(argv[1]);
  }
  if (argc == 3)
  {
    LOOP_COUNT = atoi(argv[2]);
  }
  A = (double *)mkl_malloc(n * sizeof(double), 64);
  B = (double *)mkl_malloc(n * sizeof(double), 64);
  C = (double *)mkl_malloc(n * sizeof(double), 64);
  if (A == NULL || B == NULL || C == NULL)
  {
    printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 1;
  }


  for (i = 0; i < n; i++)
  {
    A[i] = (double)(i + 1);
    B[i] = (double)(-i - 1);
  }
  s_initial = dsecnd();
  for (i = 0; i < LOOP_COUNT; i++)
  {
    vdAdd(n, A, B, C);
  }
  s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
  printf("%.5f\n", (s_elapsed * 1000));

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);

  if (s_elapsed < 0.9 / LOOP_COUNT)
  {
    s_elapsed = 1.0 / LOOP_COUNT / s_elapsed;
    i = (int)(s_elapsed * LOOP_COUNT) + 1;
    printf(" It is highly recommended to define LOOP_COUNT for this example on "
           "your \n"
           " computer as %i to have total execution time about 1 second for "
           "reliability \n"
           " of measurements\n\n",
           i);
  }

  printf(" Example completed. \n\n");
  return 0;
}
