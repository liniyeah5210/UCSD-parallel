gcc -fopenmp -O2 ParallelPoisson.c -lm -o poisson
export OMP_NUM_THREADS=12
./poisson
