// poisson.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

// Uncomment the following line to print matrices for debugging
//#define DEBUG 1

// ----------------------------------------------------------------------
// CUDA kernels for point–wise operations
// ----------------------------------------------------------------------

// Initialize f[i,j] = sin((i+1)*dx)*sin((j+1)*dx)
// f is stored as a 1D array of length N*N in row–major order.
__global__ void init_f(double *f, int N, double dx)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N)
    {
        f[i * N + j] = sin((i+1)*dx) * sin((j+1)*dx);
    }
}

// For each (i,j) compute:  c[i,j] = g[i,j] / ( - (k^2 + m^2) )
// where k = (i+1)*pi/L and m = (j+1)*pi/L.
__global__ void compute_division(const double *g, double *c, int N, double L)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N)
    {
        double k = (i+1)*M_PI / L;
        double m = (j+1)*M_PI / L;
        c[i * N + j] = g[i * N + j] / ( - (k*k + m*m) );
    }
}

// Copy the computed interior u (of size N x N) into the full solution
// (of size (N+2)x(N+2)) and set the boundaries to 0.
__global__ void copy_interior(const double *u, double *solution, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 0 <= i < N+2
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= j < N+2
    int fullSize = N + 2;
    if (i < fullSize && j < fullSize)
    {
        if (i == 0 || j == 0 || i == fullSize - 1 || j == fullSize - 1)
            solution[i * fullSize + j] = 0.0;
        else
            solution[i * fullSize + j] = u[(i-1)*N + (j-1)];
    }
}

// ----------------------------------------------------------------------
// CUDA kernels for the DST “trick” using CUFFT
// (There are two groups: one for column–transforms and one for row–transforms.)
// ----------------------------------------------------------------------

// For the DST along columns we “embed” each column of length N into an extended
// array of length M = 2*(N+1). The extended array (of type cufftDoubleComplex) is stored
// in a temporary array 'tmp'. Here each column j in the original matrix (stored in a)
// is copied into the extended array for that column.
// (Assume that the original matrix a is stored in column–major order for this kernel.)
__global__ void fill_DST_columns(const double *a, cufftDoubleComplex *tmp, int N)
{
    int j = blockIdx.x;    // one block per column (0 <= j < N)
    int i = threadIdx.x;   // thread index for rows 0<= i < N
    if (i < N && j < N)
    {
        int M = 2*(N+1);
        int idx_a = i * N + j;            // original element a(i,j)
        int idx1 = j * M + (i+1);           // fill position i+1
        int idx2 = j * M + (M - (i+1));     // fill symmetric position
        tmp[idx1].x = a[idx_a];
        tmp[idx1].y = 0.0;
        tmp[idx2].x = -a[idx_a];
        tmp[idx2].y = 0.0;
    }
}

// After the FFT, extract the DST result: for each column j and for i=0…N-1,
// set b[i,j] = factor * (imaginary part at index i+1).
// For forward DST (dir==+1), factor = -0.5.
// For inverse DST (dir==-1), factor = -0.5 * (2.0/(N+1)).
__global__ void extract_DST_columns(const cufftDoubleComplex *tmp, double *b, int N, int dir)
{
    int j = blockIdx.x;  // column index
    int i = threadIdx.x; // row index
    if (i < N && j < N)
    {
        int M = 2*(N+1);
        double factor = (dir == 1) ? -0.5 : -0.5 * (2.0/(N+1));
        // The DST result comes from the imaginary part at index (i+1)
        b[i * N + j] = factor * tmp[j * M + (i+1)].y;
    }
}

// For the DST along rows the “extended” temporary array is built for each row.
// Here the input matrix b (of size N x N, stored in row–major order) is processed
// row–by–row.
__global__ void fill_DST_rows(const double *b, cufftDoubleComplex *tmp, int N)
{
    int i = blockIdx.x;   // one block per row (0 <= i < N)
    int j = threadIdx.x;  // column index within the row
    if (i < N && j < N)
    {
        int M = 2*(N+1);
        int idx_b = i * N + j;         // element b(i,j)
        int idx1 = i * M + (j+1);        // destination index j+1
        int idx2 = i * M + (M - (j+1));  // symmetric index
        tmp[idx1].x = b[idx_b];
        tmp[idx1].y = 0.0;
        tmp[idx2].x = -b[idx_b];
        tmp[idx2].y = 0.0;
    }
}

__global__ void extract_DST_rows(const cufftDoubleComplex *tmp, double *u, int N, int dir)
{
    int i = blockIdx.x;   // row index
    int j = threadIdx.x;  // column index
    if (i < N && j < N)
    {
        int M = 2*(N+1);
        double factor = (dir == 1) ? -0.5 : -0.5 * (2.0/(N+1));
        u[i * N + j] = factor * tmp[i * M + (j+1)].y;
    }
}

// ----------------------------------------------------------------------
// Host helper functions that wrap the two passes of DST using CUFFT
// ----------------------------------------------------------------------
 
// GPU version of the first stage (columns): compute DST on each column of a.
// The result is stored in b (both stored as flattened arrays of size N*N).
// 'dir' should be +1 (forward) or -1 (inverse).
void gpu_twod_DST_columns(const double *d_a, double *d_b, int N, int dir)
{
    int M = 2*(N+1);  // extended length
    int batch = N;    // one batch per column
    size_t tmpSize = sizeof(cufftDoubleComplex) * batch * M;
    
    cufftDoubleComplex *d_tmp;
    cudaMalloc((void**)&d_tmp, tmpSize);
    // Zero out the temporary array
    cudaMemset(d_tmp, 0, tmpSize);
    
    // Launch one block per column, with N threads per block.
    fill_DST_columns<<<N, N>>>(d_a, d_tmp, N);
    cudaDeviceSynchronize();
    
    // Create a CUFFT plan for 1D complex-to-complex FFT on arrays of length M, batch = N.
    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, batch);
    // Execute the FFT (always in the forward direction).
    cufftExecZ2Z(plan, d_tmp, d_tmp, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    // Extract the DST result from the FFT output.
    extract_DST_columns<<<N, N>>>(d_tmp, d_b, N, dir);
    cudaDeviceSynchronize();
    
    cufftDestroy(plan);
    cudaFree(d_tmp);
}
 
// GPU version of the second stage (rows): compute DST on each row of b.
// The result is stored in u.
void gpu_twod_DST_rows(const double *d_b, double *d_u, int N, int dir)
{
    int M = 2*(N+1);
    int batch = N;    // one batch per row
    size_t tmpSize = sizeof(cufftDoubleComplex) * batch * M;
    
    cufftDoubleComplex *d_tmp;
    cudaMalloc((void**)&d_tmp, tmpSize);
    cudaMemset(d_tmp, 0, tmpSize);
    
    // Launch one block per row, with N threads per block.
    fill_DST_rows<<<N, N>>>(d_b, d_tmp, N);
    cudaDeviceSynchronize();
    
    cufftHandle plan;
    cufftPlan1d(&plan, M, CUFFT_Z2Z, batch);
    cufftExecZ2Z(plan, d_tmp, d_tmp, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    extract_DST_rows<<<N, N>>>(d_tmp, d_u, N, dir);
    cudaDeviceSynchronize();
    
    cufftDestroy(plan);
    cudaFree(d_tmp);
}
 
// ----------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------
int main()
{
    // Example: use N = 1023 = 2^10 - 1 so that N+1 = 1024 is a power of 2.
    int N = 8191;
    double L = M_PI;
    double dx = L / (N + 1);
    
    size_t bytesMat = sizeof(double) * N * N;
    size_t bytesSol = sizeof(double) * (N + 2) * (N + 2);
    
    // Allocate device memory for the matrices (flattened arrays)
    double *d_f, *d_g, *d_c, *d_u, *d_solution;
    cudaMalloc((void**)&d_f, bytesMat);
    cudaMalloc((void**)&d_g, bytesMat);
    cudaMalloc((void**)&d_c, bytesMat);
    cudaMalloc((void**)&d_u, bytesMat);
    cudaMalloc((void**)&d_solution, bytesSol);
    
    // Use CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // 1) Initialize f in parallel.
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);
    init_f<<<gridDim, blockDim>>>(d_f, N, dx);
    cudaDeviceSynchronize();
    
    // 2) Forward 2D DST: f -> g.
    // (This performs the DST on each column.)
    gpu_twod_DST_columns(d_f, d_g, N, +1);
    
    // 3) Divide by -((i+1)^2+(j+1)^2) in parallel: g -> c.
    compute_division<<<gridDim, blockDim>>>(d_g, d_c, N, L);
    cudaDeviceSynchronize();
    
    // 4) Inverse 2D DST: c -> u.
    // (This performs the DST on each row.)
    gpu_twod_DST_rows(d_c, d_u, N, -1);
    
    // 5) Insert boundary = 0 and copy interior u into the full solution array.
    dim3 blockDim2(16, 16);
    dim3 gridDim2(((N+2) + blockDim2.x - 1)/blockDim2.x, ((N+2) + blockDim2.y - 1)/blockDim2.y);
    copy_interior<<<gridDim2, blockDim2>>>(d_u, d_solution, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Elapsed (wall) time = %f seconds\n", elapsed_ms/1000.0);
    
    // (Optionally copy the solution back to the host and print it if DEBUG is enabled.)
#ifdef DEBUG
    double *h_solution = (double*)malloc(bytesSol);
    cudaMemcpy(h_solution, d_solution, bytesSol, cudaMemcpyDeviceToHost);
    printf("Solution (including boundaries):\n");
    int fullSize = N + 2;
    for (int i = 0; i < fullSize; i++)
    {
        for (int j = 0; j < fullSize; j++)
            printf("%.6f ", h_solution[i*fullSize + j]);
        printf("\n");
    }
    free(h_solution);
#endif
    
    // Clean up.
    cudaFree(d_f);
    cudaFree(d_g);
    cudaFree(d_c);
    cudaFree(d_u);
    cudaFree(d_solution);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
