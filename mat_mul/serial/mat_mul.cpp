
#include <bits/stdc++.h>
using namespace std;

void mat_mul(double *A, double *B, double *C, int m, int n, int p)
{
  double sum;
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      sum = 0.0;
      for (int k = 0; k < p; k++)
        C[n * i + j] += A[p * i + k] * B[n * k + j];
    }
  }
}

int main(int argc, char const *argv[])
{
  // A m*p, B p*n, C m*n
  int m, n, p, i;
  m = 1000;
  p = 1000;
  n = 1000;
  if (argc == 4)
  {
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
  clock_t start_time = clock();
  mat_mul(A, B, C, m, n, p);
  printf("time taken %f\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);
  for (int i = 0; i < min(6, m); i++)
  {
    for (int j = 0; j < min(6, p); j++)
    {
      printf("%12.5G", A[i * p + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < min(6, p); i++)
  {
    for (int j = 0; j < min(6, n); j++)
    {
      printf("%12.5G", B[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
  for (int i = 0; i < min(6, m); i++)
  {
    for (int j = 0; j < min(6, n); j++)
    {
      printf("%12.5G", C[i * n + j]);
    }
    printf("\n");
  }
  return 0;
}
