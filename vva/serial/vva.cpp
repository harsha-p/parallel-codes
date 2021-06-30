#include <bits/stdc++.h>

int main(int argc, char *argv[])
{
    double *A, *B, *C;
    int n, i, j;
    int L;
    clock_t s_initial, s_elapsed;
    n = 1000;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    if(argc >=3) L = atoi(argv[2]);
    A = (double *)malloc(n * sizeof(double));
    B = (double *)malloc(n * sizeof(double));
    C = (double *)malloc(n * sizeof(double));
    if (A == NULL || B == NULL || C == NULL)
    {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    for (i = 0; i < n; i++)
    {
        A[i] = (double)(i + 1);
        B[i] = (double)(-i - 1);
	C[i] = 0.0;
    }
    s_initial = clock();
    for(int j=0;j<L;j++){
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
    }
    s_elapsed = clock();

    printf("completed at %.5f milliseconds \n\n", ((((double)(s_elapsed - s_initial)/L)/CLOCKS_PER_SEC) * 1000));

    free(A);
    free(B);
    free(C);

    printf(" Example completed. \n\n");
    return 0;
}
