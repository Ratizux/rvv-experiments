#include "linear_algebra.h"
#include <stdio.h>
#include <time.h>

int main()
{
	const int m=300,n=301,o=302;
	Mat a=random_fill(m,n);
	Mat bt=random_fill(o,n);
	time_t time_a=clock();
	Mat ans=mat_mult(m,n,o,a,bt);
	time_t time_b=clock();
	#ifdef RVV
	Mat vvans=mat_mult_vv_m8(m,n,o,a,bt);
	time_t time_c=clock();
	#endif
	long long ops=(2*m)*(long long)n*o;
	printf("Operations: %lld\n",ops);
	double scalar_time_sec=(double)(time_b-time_a)/CLOCKS_PER_SEC;
	printf("Sum: %.1lf/Scalar",mat_sum(m,o,ans));
	#ifdef RVV
	printf(", %.1lf/Vector",mat_sum(m,o,vvans));
	#endif
	printf("\n");
	printf("Scalar: %.2lf GFLOPS\n",(double)ops/scalar_time_sec/1024/1024);
	#ifdef RVV
	double vector_time_sec=(double)(time_c-time_b)/CLOCKS_PER_SEC;
	printf("Vector: %.2lf GFLOPS\n",(double)ops/vector_time_sec/1024/1024);
	#endif
	mat_dealloc(m,a);
	mat_dealloc(o,bt);
	mat_dealloc(m,ans);
}
