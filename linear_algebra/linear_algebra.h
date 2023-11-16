#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

struct mat_f32_t
{
	int m,n;
	/*
	[1,1,4;
	 5,1,4]
	m=2, n=3
	*/
	float **mat;
};

typedef struct mat_f32_t Mat;

Mat random_fill(const int m,const int n);

//Allocate only, no initialization
Mat mat_alloc(const int m,const int n);

void mat_dealloc(const int m,Mat target);

Mat mat_copy(const int m,const int n,const Mat src);

Mat mat_mult(const int m,const int n,const int o,const Mat a,const Mat bt);

#define MAT_MULT_VV_DEF(LMUL) \
Mat mat_mult_vv_m##LMUL (const int m,const int n,const int o,const Mat a,const Mat bt);

MAT_MULT_VV_DEF(1);
MAT_MULT_VV_DEF(2);
MAT_MULT_VV_DEF(4);
MAT_MULT_VV_DEF(8);

double mat_sum(const int m,const int n,const Mat src);

#endif
