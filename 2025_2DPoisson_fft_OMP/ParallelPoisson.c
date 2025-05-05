#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>           // For OpenMP (omp_get_wtime)
#include <time.h>

// Set DEBUG to 0 to disable printing of matrices.
#define DEBUG 0

/*
   2D Fast Poisson Solver using a discrete sine transform (DST).
   (d^2/dx^2 + d^2/dy^2) u(x,y) = f(x,y), with
   f(x,y) = sin(x)*sin(y) on domain [0, pi]^2, boundary = 0.
   The analytic solution is u(x,y) = -0.5 * sin(x)*sin(y).

   We assume N = 2^k - 1, so that N+1 = 2^k.  This is needed by the
   FFT-based DST implementation.

   This version:
   - Uses OpenMP for parallelization,
   - Minimizes dynamic allocations in twod_DST(),
   - Uses omp_get_wtime() for wall-clock timing.
*/

int FFT(double *x_r, double *x_i, double *y_r, double *y_i, int N);
int DST(double *x_r, double *y_r, int N, int dir);
int twod_DST(double **a, double **b, int N, int dir);
int print_matrix(const char name[], double **a, int N);

int main()
{
    int i, j;
    // Example: N=1023 (2^10 - 1 = 1024 - 1)
    int N = 8191;
    double L = M_PI;
    double dx = L / (N + 1);

    // Check that N+1 is indeed a power of 2
    int tmp = N + 1;
    while ((tmp % 2 == 0) && tmp > 1)
        tmp /= 2;
    if (tmp != 1) {
        printf("ERROR: N must be 2^k - 1\n");
        return 1;
    }

    // Allocate arrays
    double **f = (double **) malloc(N * sizeof(double*));
    f[0]       = (double *) malloc(N * N * sizeof(double));
    for (i = 1; i < N; i++)
        f[i] = f[0] + i*N;

    double **g = (double **) malloc(N * sizeof(double*));
    g[0]       = (double *) malloc(N * N * sizeof(double));
    for (i = 1; i < N; i++)
        g[i] = g[0] + i*N;

    double **c = (double **) malloc(N * sizeof(double*));
    c[0]       = (double *) malloc(N * N * sizeof(double));
    for (i = 1; i < N; i++)
        c[i] = c[0] + i*N;

    double **u = (double **) malloc(N * sizeof(double*));
    u[0]       = (double *) malloc(N * N * sizeof(double));
    for (i = 1; i < N; i++)
        u[i] = u[0] + i*N;

    // This will store the final solution, including boundaries
    double **solution = (double **) malloc((N+2) * sizeof(double*));
    solution[0]       = (double *) malloc((N+2)*(N+2)*sizeof(double));
    for (i = 1; i < N+2; i++)
        solution[i] = solution[0] + i*(N+2);

#if DEBUG
    printf("2D Poisson with DST, N = %d\n", N);
    printf("(d^2/dx^2 + d^2/dy^2)u = sin(x)*sin(y)\n");
    printf("Analytic = -0.5 * sin(x)*sin(y)\n");
#endif

    double t_start = omp_get_wtime();

    // 1) Initialize f in parallel
    #pragma omp parallel for private(i,j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            f[i][j] = sin((i+1)*dx) * sin((j+1)*dx);
        }
    }

#if DEBUG
    print_matrix("f", f, N);
#endif

    // 2) Forward 2D DST: f -> g
    twod_DST(f, g, N, /*dir=*/ +1);

#if DEBUG
    print_matrix("g", g, N);
#endif

    // 3) Divide by -((i+1)^2 + (j+1)^2) in parallel => c
    #pragma omp parallel for private(i,j)
    for (i = 0; i < N; i++) {
        double k = (i+1)*M_PI / L;
        for (j = 0; j < N; j++) {
            double m = (j+1)*M_PI / L;
            c[i][j] = g[i][j] / ( - (k*k + m*m) );
        }
    }

#if DEBUG
    print_matrix("c", c, N);
#endif

    // 4) Inverse 2D DST: c -> u
    twod_DST(c, u, N, /*dir=*/ -1);

    // 5) Insert boundary = 0, copy interior to solution
    #pragma omp parallel for private(i,j)
    for (i = 0; i < N+2; i++) {
        // Boundaries in top/bottom row
        solution[0][i]   = 0.0;
        solution[N+1][i] = 0.0;
        // Boundaries in left/right column
        solution[i][0]   = 0.0;
        solution[i][N+1] = 0.0;
    }

    #pragma omp parallel for private(i,j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            solution[i+1][j+1] = u[i][j];
        }
    }

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

#if DEBUG
    // Print final solution
    print_matrix("solution", solution, N+2);

    // Compare to analytic
    printf("Analytic:\n");
    for (i = 0; i < N+2; i++) {
        for (j = 0; j < N+2; j++) {
            double val = -0.5 * sin(i * dx) * sin(j * dx);
            printf("%.6f ", val);
        }
        printf("\n");
    }
#endif

    printf("Elapsed (wall) time = %f seconds\n", elapsed);

    // Free memory
    free(f[0]); free(f);
    free(g[0]); free(g);
    free(c[0]); free(c);
    free(u[0]); free(u);
    free(solution[0]); free(solution);

    return 0;
}


/*
 * 2D DST:
 *
 * We do the DST in two stages:
 *   (a) DST of columns:   a -> b
 *   (b) DST of rows (in-place on b):  b -> b
 *
 * This avoids extra "tmp" arrays and large repeated allocations.
 * 'dir' = +1 means forward DST, -1 means inverse DST.
 */
int twod_DST(double **a, double **b, int N, int dir)
{
    // (1) DST each column: read from a, write to b
    #pragma omp parallel
    {
        double *x = (double *) malloc(N * sizeof(double));
        double *y = (double *) malloc(N * sizeof(double));
        #pragma omp for
        for (int j = 0; j < N; j++) {
            // copy column j from "a" into x
            for (int i = 0; i < N; i++) {
                x[i] = a[i][j];
            }
            // 1D DST transform
            DST(x, y, N, dir);
            // store in column j of b
            for (int i = 0; i < N; i++) {
                b[i][j] = y[i];
            }
        }
        free(x);
        free(y);
    }

    // (2) DST each row of b, in place
    #pragma omp parallel
    {
        double *x = (double *) malloc(N * sizeof(double));
        double *y = (double *) malloc(N * sizeof(double));
        #pragma omp for
        for (int i = 0; i < N; i++) {
            // copy row i from b
            for (int j = 0; j < N; j++) {
                x[j] = b[i][j];
            }
            // 1D DST
            DST(x, y, N, dir);
            // copy back into row i of b
            for (int j = 0; j < N; j++) {
                b[i][j] = y[j];
            }
        }
        free(x);
        free(y);
    }

    return 0;
}

/*
 * 1D Discrete Sine Transform using a trick that calls FFT
 * dir = +1 => forward DST
 * dir = -1 => inverse DST
 */
int DST(double *x_r, double *y_r, int N, int dir)
{
    if (dir != 1 && dir != -1) {
        printf("ERROR: DST direction must be +1 or -1\n");
        return 1;
    }

    // We'll allocate arrays for the “FFTW trick”
    double *x2_r = (double *) calloc(2*(N+1), sizeof(double));
    double *x2_i = (double *) calloc(2*(N+1), sizeof(double));
    double *y2_r = (double *) calloc(2*(N+1), sizeof(double));
    double *y2_i = (double *) calloc(2*(N+1), sizeof(double));

    // Populate x2_r for DST
    for (int i=0; i<N; i++) {
        x2_r[i+1] = x_r[i];
        x2_r[2*(N+1) - (i+1)] = -x_r[i];
    }

    // Compute FFT
    FFT(x2_r, x2_i, y2_r, y2_i, 2*(N+1));

    // Extract results
    if (dir == 1) {  // forward DST
        for (int i=0; i<N; i++) {
            // -0.5 * imag( FFT(...) )   from index i+1
            y_r[i] = -0.5 * y2_i[i+1];
        }
    } else {  // inverse DST
        for (int i=0; i<N; i++) {
            // -0.5 * imag( FFT(...) ) * 2/(N+1)
            y_r[i] = -0.5 * y2_i[i+1] * 2.0 / (N+1);
        }
    }

    free(x2_r);
    free(x2_i);
    free(y2_r);
    free(y2_i);
    return 0;
}

/*
 * 1D FFT (recursive Cooley-Tukey)
 * x_r + i x_i => y_r + i y_i
 * N must be a power of 2.
 */
int FFT(double *x_r, double *x_i, double *y_r, double *y_i, int N)
{
    if (N == 2) {
        // Base case
        y_r[0] = x_r[0] + x_r[1];
        y_i[0] = x_i[0] + x_i[1];
        y_r[1] = x_r[0] - x_r[1];
        y_i[1] = x_i[0] - x_i[1];
        return 0;
    }

    int half = N/2;
    double *z_r = (double *) malloc(N*sizeof(double));
    double *z_i = (double *) malloc(N*sizeof(double));
    double *u_r = (double *) malloc(N*sizeof(double));
    double *u_i = (double *) malloc(N*sizeof(double));

    // De-interleave
    for (int k=0; k<half; k++) {
        z_r[k]       = x_r[2*k];
        z_i[k]       = x_i[2*k];
        z_r[k+half]  = x_r[2*k+1];
        z_i[k+half]  = x_i[2*k+1];
    }

    // Recursively FFT even indices => (u_r,u_i)
    FFT(z_r, z_i, u_r, u_i, half);
    // Recursively FFT odd indices => (u_r+half,u_i+half)
    FFT(z_r+half, z_i+half, u_r+half, u_i+half, half);

    // Combine
    double w_N_re = cos(2.0*M_PI / N);
    double w_N_im = -sin(2.0*M_PI / N);
    double w_re   = 1.0;
    double w_im   = 0.0;

    for (int k=0; k<half; k++) {
        double a = w_re*u_r[half+k] - w_im*u_i[half+k];
        double b = w_re*u_i[half+k] + w_im*u_r[half+k];

        y_r[k]       = u_r[k] + a;
        y_i[k]       = u_i[k] + b;
        y_r[k+half]  = u_r[k] - a;
        y_i[k+half]  = u_i[k] - b;

        // Update twiddle factor
        double tmp  = w_re;
        w_re        = w_re*w_N_re - w_im*w_N_im;
        w_im        = tmp*w_N_im + w_im*w_N_re;
    }

    free(z_r);
    free(z_i);
    free(u_r);
    free(u_i);
    return 0;
}

int print_matrix(const char name[], double **a, int N)
{
    printf("%s =\n", name);
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            printf("%.6f ", a[i][j]);
        }
        printf("\n");
    }
    return 0;
}
