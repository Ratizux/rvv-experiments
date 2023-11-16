#include "linear_algebra.h"
#include <stdlib.h>
#include <time.h>
//This implementation assumes memory is sufficient

Mat random_fill(const int m,const int n)
{
	Mat target=mat_alloc(m,n);
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++) target.mat[i][j]=(float)rand()/(float)RAND_MAX;//0..1 random value
	}
	return target;
}

Mat mat_alloc(const int m,const int n)
{
	Mat target;
	target.m=m;
	target.n=n;
	target.mat=(float**)calloc(m,sizeof(float*));
	for(int i=0;i<m;i++) target.mat[i]=(float*)calloc(n,sizeof(float));
	return target;
}

void mat_dealloc(const int m,Mat target)
{
	for(int i=0;i<m;i++) free(target.mat[i]);
	free(target.mat);
}

Mat mat_copy(const int m,const int n,const Mat src);

#ifdef RVV

#include <riscv_vector.h>

//Full description available in matrix.c
#define MAT_MULT_VV(LMUL) \
Mat mat_mult_vv_m##LMUL (const int m,const int n,const int o,const Mat a,const Mat bt) \
{ \
	Mat ans=mat_alloc(m,o); \
	size_t vl; \
	size_t vlmax=vsetvlmax_e32m##LMUL(); \
	vfloat32m##LMUL##_t vzero,vans; \
	vzero=vfmv_v_f_f32m##LMUL(0.0,vlmax); \
	for(int i=0;i<m;i++) \
	{ \
		for(int j=0;j<o;j++) \
		{ \
			vans=vzero; \
			float *pa=a.mat[i],*pbt=bt.mat[j]; \
			for(size_t length=n;length>0;length-=vl) \
			{ \
				vfloat32m##LMUL##_t va,vb; \
				vl=vsetvl_e32m##LMUL(length); \
				if(vl<vlmax) \
				{ \
					va=vzero; \
					vb=vzero; \
				} \
				va=vle32_v_f32m##LMUL(pa,vl); \
				vb=vle32_v_f32m##LMUL(pbt,vl); \
				vans=vfmacc_vv_f32m##LMUL(vans,va,vb,vlmax); \
				pa+=vl; \
				pbt+=vl; \
			} \
			vfloat32m1_t vsum; \
			vsum=vfredusum_vs_f32m##LMUL##_f32m1(vsum,vans,vfmv_v_f_f32m1(0.0,vlmax),vlmax); \
			ans.mat[i][j]=vfmv_f_s_f32m1_f32(vsum); \
		} \
	} \
	return ans; \
}

MAT_MULT_VV(1)
MAT_MULT_VV(2)
MAT_MULT_VV(4)
MAT_MULT_VV(8)

#endif
Mat mat_mult(const int m,const int n,const int o,const Mat a,const Mat bt)
{
	Mat ans=mat_alloc(m,o);
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<o;j++)
		{
			ans.mat[i][j]=0;
			for(int k=0;k<n;k++) ans.mat[i][j]+=a.mat[i][k]*bt.mat[j][k];
		}
	}
	return ans;
}

double mat_sum(const int m,const int n,const Mat src)
{
	double ans=0;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			ans+=src.mat[i][j];
		}
	}
	return ans;
}
