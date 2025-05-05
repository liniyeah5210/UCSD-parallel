#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#include <omp.h>
#define DEBUG 0
#define PRINT_SOLUTION 0

/*
2D fast Poisson solver. 
(d^2/dx^2)u(x,y) + (d^2/dy^2)u(x,y) = f(x,y)
f(x,y) = sin(x)*sin(y)
domain = [0,pi]*[0*pi]
Boundary points are all zeros. 
Analytic solution: -0.5*sin(x)*sin(y)
I use DST-I from wiki.
*/

int FFT(double *x_r, double *x_i, double *y_r, double *y_i, int N);
int DST(double *x_r, double *y_r, int N, int dir);
int twod_DST(double **a, double **b, int N, int dir);
int print_matrix(char name[], double **a, int N);

int main()
{
	int i, j, x, N, print, dir;
	clock_t t1, t2;
	double **f, **g, **c, **u, **mesh, **solution;
	double L, dx, err, errij, start, end;

	N=15; //Number of interior grids, must be 2^k-1 (for fft computation)
	L=M_PI; // domain size is [0,pi]*[0,pi]	
	dx=L/(N+1);

	//Check if N is 2^k-1	
	x=N+1;
	while (((x % 2) == 0) && x > 1) x /= 2; 
	if(x!=1)
	{
	printf("N must be must be 2^k-1 (for fft computation)\n");
	return 1;
	}
	

	f = (double **) malloc( N * sizeof(double *)); //f does not include boundary points  
	f[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) f[i] = f[0] + i*N;


	g = (double **) malloc( N * sizeof(double *)); //g = twod_DST(f) 
	g[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) g[i] = g[0] + i*N;


	c = (double **) malloc( N * sizeof(double *)); //c = g/(-k^2-m^2) 
	c[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) c[i] = c[0] + i*N;

	u = (double **) malloc( N * sizeof(double *)); //solution without boundary points 
	u[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) u[i] = u[0] + i*N;

	solution = (double **) malloc((N+2) * sizeof(double *)); //solution with boundary points 
	solution[0] = (double *) malloc((N+2)*(N+2)*sizeof(double));
	for(i=1;i<N+2;++i) solution[i] = solution[0] + i*(N+2);



	printf("(d^2/dx^2)u(x,y) + (d^2/dy^2)u(x,y) = f(x,y)\n");
	printf("f(x,y) = sin(x)*sin(y)\n");
	printf("domain = [0,pi]*[0*pi] \n");
	printf("Boundary points are all zeros.\n\n");
	printf("Analytic solution: u(x,y) = -0.5*sin(x)*sin(y) \n");
	printf("Number of interior grids: %d  \n", N+1);




        # pragma omp parallel\
	  shared(N, f, dx,)\
	  private(i, j)
	{
        # pragma omp for nowait
	for(i=0;i<N;++i) 
	{
		for(j=0;j<N;++j) f[i][j]=sin((i+1)*dx)*sin((j+1)*dx);
	}
	}




	#if DEBUG	
	print_matrix("f",f,N);
	#endif


       start = omp_get_wtime();



	
	twod_DST(f,g,N,1);
	

	#if DEBUG	
	print_matrix("g",g,N);
	#endif
	

        # pragma omp parallel shared(c,g,L,N) private(i,j)
        {
         # pragma omp for nowait
	for(i=0;i<N;++i)
	{
		for(j=0;j<N;++j)
		{
			c[i][j] = g[i][j]/(-pow(((i+1)*M_PI/L),2)-pow(((j+1)*M_PI/L),2));
		}
	}

	}



	#if DEBUG
	print_matrix("c",c,N);
	#endif


	
	twod_DST(c,u,N,-1);



       end = omp_get_wtime();



	for(i=0;i<N+2;++i)
	{
		solution[0][i] = 0;
		solution[N+1][i] = 0;
		solution[i][0] = 0;
		solution[i][N+1] = 0;
	}



	for(i=0;i<N;++i)
	{
		for(j=0;j<N;++j)
		{
			solution[i+1][j+1] = u[i][j];
		}
	}



	#if PRINT_SOLUTION
	print_matrix("numerical solution",solution,N+2);


	printf("analytical soluion = \n");
	for(i=0;i<N+2;++i)
	{
		for(j=0;j<N+2;++j)
		{
			printf("%lf ", -0.5*sin(i*dx)*sin(j*dx));
		}
		printf("\n");
	
	}
	#endif



	err=0;
	for(i=0;i<N+2;++i)
	{
		for(j=0;j<N+2;++j)
		{
		  errij=solution[i][j] -(-0.5*sin(i*dx)*sin(j*dx));
		  err=err+errij;
		}
	}
	
	err=err/((N+2)*(N+2));
	printf("average arror = %e \n ",err);

        printf("Work took %lf seconds\n", end-start);



	return 0;

}



int print_matrix(char name[], double **a, int N)
{
	int i, j;

	printf("%s = \n", name);
	for(i=0;i<N;++i)
	{
		for(j=0;j<N;++j)
		{
			printf("%lf ",a[i][j]);
		}
		printf("\n");
	}



}



int twod_DST(double **a, double **b, int N, int dir)
{
	int i, j;
	double **tmp;
	double *x, *y;

	x = (double *) malloc( N * sizeof(double));
	y = (double *) malloc( N * sizeof(double));

	tmp = (double **) malloc( N * sizeof(double *));  
	tmp[0] = (double *) malloc(N*N*sizeof(double));
	for(i=1;i<N;++i) tmp[i] = tmp[0] + i*N;

	for(j=0;j<N;j++)
	{
		for(i=0;i<N;i++) x[i] = a[i][j];
		DST(x,y,N,dir);	
		for(i=0;i<N;i++) tmp[i][j] = y[i];

	}

	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++) x[j] = tmp[i][j];
		DST(x,y,N,dir);
		for(j=0;j<N;j++) b[i][j] = y[j];
	}

}






int DST(double *x_r, double *y_r, int N, int dir)
{

	int i;
	double *x2_r,*x2_i,*y2_r,*y2_i;
	x2_r = (double *) malloc(2*(N+1) * sizeof(double));
	x2_i = (double *) malloc(2*(N+1) * sizeof(double));
	y2_r = (double *) malloc(2*(N+1) * sizeof(double));
	y2_i = (double *) malloc(2*(N+1) * sizeof(double));
	

	if(dir!=1 && dir!=-1)
	{
	printf("dir must be +1 or -1\n");
	return 1;
	}

	for(i=0;i<2*(N+1);++i)
	{
		x2_r[i] = 0;
		x2_i[i] = 0;
		y2_r[i] = 0;
		y2_i[i] = 0;
	}
	
        for(i=0;i<N;++i)
	{
		x2_r[i+1] = x_r[i]; 
		x2_r[2*(N+1)-i-1] = -x_r[i];
	}
	

	FFT(x2_r, x2_i, y2_r, y2_i, 2*(N+1));


        # pragma omp parallel\
	  shared(N, y_r, y2_i)\
	  private(i)
	{
	if(dir==1)
        # pragma omp for nowait
	for(i=0;i<N;++i)
	{
		y_r[i] = -0.5*y2_i[i+1];
        }


	if(dir==-1)
        # pragma omp for nowait
	for(i=0;i<N;++i)
	{
		y_r[i] = -0.5*y2_i[i+1]*2/(N+1);
	}


	}


}




int FFT(double *x_r, double *x_i, double *y_r, double *y_i, int N)
{
	int k;
	double w_N_re, w_N_im, temp, u1_r, u1_i, u2_r, u2_i, a, b, c, d;
	double *z_r, *z_i, *u_r, *u_i, *w_re, *w_im ;

	if(N == 2)
	{
	y_r[0] = x_r[0] + x_r[1];
	y_i[0] = x_i[0] + x_i[1];
	y_r[1] = x_r[0] - x_r[1]; 
	y_i[1] = x_i[0] - x_i[1];
	} 



	else
	{
	double *z_r, *z_i, *u_r, *u_i, *w_re, *w_im;
	w_re = (double *) malloc(N/2*sizeof(double)); 
	w_im = (double *) malloc(N/2*sizeof(double)); 
	

	z_r = (double *) malloc(N*sizeof(double)); // z_r:0~N/2-1: even, z_r:N/2~N-1: odd
	if(z_r==NULL) { printf("no memory!!\n"); return 0; }
	z_i = (double *) malloc(N*sizeof(double)); // z_i:0~N/2-1: even, z_i:N/2~N-1: odd 
	if(z_i==NULL) { printf("no memory!!\n"); return 0; }
	u_r = (double *) malloc(N*sizeof(double)); // z_r:0~N/2-1: even, z_r:N/2~N-1: odd
	if(u_r==NULL) { printf("no memory!!\n"); return 0; }
	u_i = (double *) malloc(N*sizeof(double)); // z_i:0~N/2-1: even, z_i:N/2~N-1: odd 
	if(u_i==NULL) { printf("no memory!!\n"); return 0; }
	for(k=0;k<N/2;++k)
	{
		z_r[k] = x_r[2*k];
		z_i[k] = x_i[2*k];
		z_r[N/2+k]  = x_r[2*k+1];
		z_i[N/2+k]  = x_i[2*k+1];
	}		
	FFT(z_r, z_i, u_r, u_i, N/2);
	FFT(z_r+N/2, z_i+N/2, u_r+N/2, u_i+N/2, N/2);
	w_N_re =  cos(2.0*M_PI/N);
	w_N_im = -sin(2.0*M_PI/N);
	w_re[0] = 1.0;
	w_im[0] = 0.0; 

	for(k=0;k<N/2;++k)
	{
	w_re[k+1] = w_re[k]*w_N_re - w_im[k]*w_N_im;
	w_im[k+1] = w_re[k]*w_N_im + w_im[k]*w_N_re;
	}


        # pragma omp parallel\
	  shared(N, w_re ,w_im, y_r, y_i, w_N_re, w_N_im, u_r, u_i)\
	  private(k, a, b)
	{
        # pragma omp for nowait
	for(k=0;k<N/2;++k)
	{
		a = w_re[k]*u_r[N/2+k] - w_im[k]*u_i[N/2+k];
		b = w_re[k]*u_i[N/2+k] + w_im[k]*u_r[N/2+k];
		y_r[k]     = u_r[k] + a;
		y_i[k]     = u_i[k] + b;
		y_r[N/2+k] = u_r[k] - a;
		y_i[N/2+k] = u_i[k] - b;
	}
	}

	free(u_r); free(u_i); free(z_r); free(z_i);
	}	
	





	return 0;
}
