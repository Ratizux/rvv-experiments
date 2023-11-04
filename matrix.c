#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define RVV

#ifdef RVV
#include <riscv_vector.h>
#endif

float** empty_fill(int m,int n)
{
	float **target;
	target=(float**)calloc(m,sizeof(float*));
	for(int i=0;i<m;i++) target[i]=(float*)calloc(n,sizeof(float*));
	return target;
}

float** random_fill(int m,int n)
{
	float **target;
	target=(float**)calloc(m,sizeof(float*));
	for(int i=0;i<m;i++)
	{
		target[i]=(float*)calloc(n,sizeof(float*));
		for(int j=0;j<n;j++)
		{
			target[i][j]=(float)rand()/(float)RAND_MAX+(float)(rand()%100);
		}
	}
	return target;
}

float** destroy(float **target,int m)
{
	for(int i=0;i<m;i++) free(target[i]);
	free(target);
	return NULL;
}

void matrix_mtpl(float **a,float **bt,float **ans,int m,int n,int o)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<o;j++)
		{
			ans[i][j]=0;
			for(int k=0;k<n;k++) ans[i][j]+=a[i][k]*bt[j][k];
		}
	}
}

#ifdef RVV
void rvv_matrix_mtpl(float **a,float **bt,float **ans,int m,int n,int o)
{
	size_t vl;//和上面的 vl 含义一致
	size_t vlmax=vsetvlmax_e32m1();//相当于下面用到的向量寄存器里，最多存储的元素个数
	vfloat32m1_t vzero,vans;
	vzero=vfmv_v_f_f32m1(0.0,vlmax);//设置一个全 0.0 填充的向量寄存器，下面有用
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<o;j++)
		{
			vans=vzero;//vans 存储 a[i][0]*bt[j][0]..a[i][n]*bt[j][n] 的和
			//计算开始前将 vans 置零
			float *pa=a[i],*pbt=bt[j];//指针指向将要被计算的第一个元素
			//初始状态 pa 指向 a[i][0]，pbt 指向 bt[j][0]
			for(size_t length=n;length>0;length-=vl)
			{
				vfloat32m1_t va,vb;
				vl=vsetvl_e32m1(length);//同样是一次处理 vl 个元素
				if(vl<vlmax)
				{
					va=vzero;//给 va 置零
					vb=vzero;
				}
				va=vle32_v_f32m1(pa,vl);//va 存储 a[i][x],a[i][x+1],...
				vb=vle32_v_f32m1(pbt,vl);//vb 存储 bt[j][x],bt[j][x+1],...
				//如果 vl < vlmax，剩下的没被设定的部分是 0
				//最新 rvv-intrinsic-doc/examples 是没有给 va vb 置零这一步的，可能是因为新版 vfmacc_vv 的行为不太一样
				vans=vfmacc_vv_f32m1(vans,va,vb,vlmax);//向量乘加的操作
				//最新 rvv-intrinsic-doc/examples 里 vfmacc_vv 最后一个参数是 vl
				//我这里如果设成 vl，且 vl < vlmax 的话，vans 后面的数据似乎会丢失
				//所以把操作元素数改成了 vlmax。前面的 va vb 置零是为了保证没有用到的部分是 0
				//
				//此处 va 中的每个元素分别与 vb 中同一下标的元素相乘
				//得到的向量再与 vans 相加
				//如果写成 Octave 语法，是 vans=vans+va*vb
				//如果vl=4，可以看出 vans[0] 最终会等于 a[i][0]*bt[j][0] + a[i][4]*bt[j][4] + ...
				pa+=vl;//设置偏移量
				pbt+=vl;
			}
			vfloat32m1_t vsum;
			vsum=vfredusum_vs_f32m1_f32m1(vsum,vans,vzero,vlmax);//求向量vans中各个元素的和（得到标量），再与vzero[0]（标量，这里是0）相加，写入vsum[0]
			//这就得到 a[i][0]*bt[j][0] + a[i][1]*bt[j][1] + ... 的和
			//vfredusum_vs 的第一个参数我也不知道是什么意思，实际上好像填什么都不影响运算
			//最新 RVV Intrinsic 里是没有第一个参数的
			//有些代码在第一个参数处写即将被覆盖的寄存器，所以我也这么写了
			ans[i][j]=vfmv_f_s_f32m1_f32(vsum);//取出 vsum[0]，写入内存
		}
	}
}

void rvv_matrix_mtpl_m8(float **a,float **bt,float **ans,int m,int n,int o)
{
	size_t vl;
	size_t vlmax=vsetvlmax_e32m8();
	vfloat32m8_t vzero,vans;
	vfloat32m1_t vzero_m1;
	vzero=vfmv_v_f_f32m8(0.0,vlmax);
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<o;j++)
		{
			vans=vzero;
			float *pa=a[i],*pbt=bt[j];
			for(size_t length=n;length>0;length-=vl)
			{
				vfloat32m8_t va,vb;
				vl=vsetvl_e32m8(length);
				if(vl<vlmax)
				{
					va=vzero;
					vb=vzero;
				}
				va=vle32_v_f32m8(pa,vl);
				vb=vle32_v_f32m8(pbt,vl);
				vans=vfmacc_vv_f32m8(vans,va,vb,vlmax);
				pa+=vl;
				pbt+=vl;
			}
			vfloat32m1_t vsum;
			vsum=vfredusum_vs_f32m8_f32m1(vsum,vans,vzero_m1,vlmax);
			ans[i][j]=vfmv_f_s_f32m1_f32(vsum);
		}
	}
}
#endif

void matrix_print(float **target,int m,int n)
{
	printf("[");
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			printf("%f",target[i][j]);
			if(j!=n-1) printf(",");
		}
		if(i!=m-1)printf(";\n");
	}
	printf("]\n\n");
}

void matrix_uncertainty(float **target,float **standard,int m,int n)
{
	printf("[");
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			printf("%f",fabs(standard[i][j]-target[i][j])/standard[i][j]);
			if(j!=n-1) printf(",");
		}
		if(i!=m-1)printf(";\n");
	}
	printf("]\n\n");
}

void benchmark(int n,int dim_min,int dim_max)
{
	if(dim_min<1) return;
	if(dim_max>100) return;
	if(dim_max<dim_min) return;
	if(n<1) return;
	if(n>100) return;
	srand((unsigned)time(NULL));
	float ***a,***bt,***ans,***ansv;
	int *dim_m,*dim_n,*dim_o;
	a=(float***)calloc(n,sizeof(float**));
	bt=(float***)calloc(n,sizeof(float**));
	ans=(float***)calloc(n,sizeof(float**));
	ansv=(float***)calloc(n,sizeof(float**));
	dim_m=(int*)calloc(n,sizeof(int));
	dim_n=(int*)calloc(n,sizeof(int));
	dim_o=(int*)calloc(n,sizeof(int));
	for(int i=0;i<n;i++)
	{
		dim_m[i]=dim_min+rand()%(dim_max-dim_min+1);//rand between dim_min and dim_max
		dim_n[i]=dim_min+rand()%(dim_max-dim_min+1);
		dim_o[i]=dim_min+rand()%(dim_max-dim_min+1);
		a[i]=random_fill(dim_m[i],dim_n[i]);
		bt[i]=random_fill(dim_o[i],dim_n[i]);
		ans[i]=empty_fill(dim_m[i],dim_o[i]);
		ansv[i]=empty_fill(dim_m[i],dim_o[i]);
	}
	clock_t time_a,time_b,time_c;
	time_a=clock();
	for(int i=0;i<n;i++) matrix_mtpl(a[i],bt[i],ans[i],dim_m[i],dim_n[i],dim_o[i]);//matrix fn
	time_b=clock();
	for(int i=0;i<n;i++) rvv_matrix_mtpl(a[i],bt[i],ansv[i],dim_m[i],dim_n[i],dim_o[i]);//matrix fn
	time_c=clock();

	int passed=1;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<dim_m[i];j++)
		{
			for(int k=0;k<dim_o[i];k++)
			{
				if((fabs(ans[i][j][k]-ansv[i][j][k])/ans[i][j][k])>0.00001)
				{
					passed=0;
					break;
				}
			}
		}
	}
	printf("dim_min: %d\ndim_max: %d\n",dim_min,dim_max);
	if(passed) printf("test passed\nno intrinsic: %ld\n   intrinsic: %ld\n",time_b-time_a,time_c-time_b);
	else printf("test failed\n");
}

int main()
{
	benchmark(50,10,20);
	benchmark(50,20,50);
	benchmark(50,50,100);
}
