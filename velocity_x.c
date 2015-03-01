#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define align 0
#define iblock 30
#define jblock 5
#define kblock 510
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

void dvelcx(float DH, float DT,
			int nxt, int nyt, int nzt,
			float* ptr_u1, float* ptr_v1, float* ptr_w1,
			float* ptr_xx, float* ptr_yy, float* ptr_zz, float* ptr_xy, float* ptr_xz, float* ptr_yz,
			float* ptr_dcrjx, float* ptr_dcrjy, float* ptr_dcrjz, float* ptr_d1, int i_s, int j_s, int k_s)
{
	float c1, c2;
	float dth;
	int   slice_1,  slice_2,  yline_1,  yline_2;
	int i_e = MIN(i_s+iblock-1, nxt+3);
	int j_e = MIN(j_s+jblock-1, nyt-1);
	int k_e = MIN(k_s+kblock-1, nzt+align-1);
	int ii, jj, kk;

	slice_1  = (nyt+8)*(nzt+2*align);
	slice_2  = (nyt+8)*(nzt+2*align)*2;
	yline_1  = nzt+2*align;
	yline_2  = (nzt+2*align)*2;

	c1 = 9.0/8.0;
	c2  = -1.0/24.0;
	dth = DT/DH;
	
	for (ii = i_s; ii <= i_e; ii++)
	{
		float dcrjx = ptr_dcrjx[ii];
		float *d1 = &ptr_d1[ii*slice_1+j_s*yline_1];
		float *u1 = &ptr_u1[ii*slice_1+j_s*yline_1];
		float *v1 = &ptr_v1[ii*slice_1+j_s*yline_1];
		float *w1 = &ptr_w1[ii*slice_1+j_s*yline_1];
		float *xx = &ptr_xx[ii*slice_1+j_s*yline_1];
		float *yy = &ptr_yy[ii*slice_1+j_s*yline_1];
		float *zz = &ptr_zz[ii*slice_1+j_s*yline_1];
		float *xy = &ptr_xy[ii*slice_1+j_s*yline_1];
		float *xz = &ptr_xz[ii*slice_1+j_s*yline_1];
		float *yz = &ptr_yz[ii*slice_1+j_s*yline_1];

		for (jj = j_s; jj <= j_e; jj++, d1+=yline_1, u1+=yline_1, v1+=yline_1, w1+=yline_1, xx+=yline_1, yy+=yline_1, zz+=yline_1, xy+=yline_1, xz+=yline_1, yz+=yline_1)
		{
			float dcrjy = ptr_dcrjy[jj];
			#pragma vector always
			#pragma simd
			for (kk = k_s; kk <= k_e; kk++)
			{
				register int dcrj;
				register int d_1, d_2, d_3;

				dcrj = dcrjx * dcrjy * ptr_dcrjz[kk];
				d_1 = 0.25*((d1[kk]+d1[kk-yline_1])+(d1[kk-1]+d1[kk-yline_1-1]));
				d_2 = 0.25*((d1[kk]+d1[kk+slice_1])+(d1[kk-1]+d1[kk+slice_1-1]));
				d_3 = 0.25*((d1[kk]+d1[kk+slice_1])+(d1[kk-yline_1]+d1[kk+slice_1-yline_1]));

				u1[kk] = (u1[kk] + (dth/d_1)*(c1*(xx[kk]-xx[kk-slice_1]) + c2*(xx[kk+slice_1]-xx[kk-slice_2])
											+ c1*(xy[kk]-xy[kk-yline_1]) + c2*(xy[kk+yline_1]-xy[kk-yline_2])
											+ c1*(xz[kk]-xz[kk-1]) + c2*(xz[kk+1]-xz[kk-2]))) * dcrj;

				v1[kk] = (v1[kk] + (dth/d_2)*(c1*(xy[kk+slice_1]-xy[kk]) + c2*(xy[kk+slice_2]-xy[kk-slice_1])
											+ c1*(yy[kk+yline_1]-yy[kk]) + c2*(yy[kk+yline_2]-yy[kk-yline_1])
											+ c1*(yz[kk]-yz[kk-1]) + c2*(yz[kk+1]-yz[kk-2]))) * dcrj;

				w1[kk] = (w1[kk] + (dth/d_3)*(c1*(xz[kk+slice_1]-xz[kk]) + c2*(xz[kk+slice_2]-xz[kk-slice_1])
											+ c1*(yz[kk]-yz[kk-yline_1]) + c2*(yz[kk+yline_1]-yz[kk-yline_2])
											+ c1*(zz[kk+1]-zz[kk]) + c2*(zz[kk+2]-zz[kk-1]))) * dcrj;
			}
		}
	}


	return;
}

void dvelcx_omp(float DH, float DT,
			int nxt, int nyt, int nzt,
			float* u1, float* v1, float* w1,
			float* xx, float* yy, float* zz, float* xy, float* xz, float* yz,
            float* dcrjx, float* dcrjy, float* dcrjz, float *d1)
{
	int i, j, k;
	int index = 0;
	int num_blocks, num_i_blocks, num_j_blocks, num_k_blocks;
	int *blocking;

	num_i_blocks = ceil((double)nxt/iblock);
	num_j_blocks = ceil((double)(nyt-8)/jblock);
	num_k_blocks = ceil((double)nzt/kblock);
	num_blocks = num_i_blocks * num_j_blocks * num_k_blocks;


	blocking = (int*)malloc(3*num_blocks*sizeof(int));
	if (blocking == NULL) printf ("Allocate blocking failed!\n");

	for (i = 4; i <= nxt+3; i += iblock)
		for (j = 8; j <= nyt-1; j += jblock)
			for (k = align; k <= nzt+align-1; k += kblock)
			{
				blocking[index++] = i;
				blocking[index++] = j;
				blocking[index++] = k;
			}

	#pragma omp parallel for schedule(dynamic) firstprivate (nxt, nyt, nzt, num_blocks) shared (DH, DT, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, blocking)
	for (i = 0; i < num_blocks; i++)
	{
		int i_s = blocking[3*i];
		int j_s = blocking[3*i+1];
		int k_s = blocking[3*i+2];	
		dvelcx(DT, DH, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, i_s, j_s, k_s);
	}

	free(blocking);
	return;
}

void dvelcx_omp_2(float DH, float DT,
		int nxt, int nyt, int nzt,
		float* u1, float* v1, float* w1,
		float* xx, float* yy, float* zz, float* xy, float* xz, float* yz,
        float* dcrjx, float* dcrjy, float* dcrjz, float *d1)
{

	float c1, c2;
	float dth;
	int slice_1,  slice_2,  yline_1,  yline_2;
	int i, j, k;

	slice_1  = (nyt+8)*(nzt+2*align);
	slice_2  = (nyt+8)*(nzt+2*align)*2;
	yline_1  = nzt+2*align;
	yline_2  = (nzt+2*align)*2;

	c1 = 9.0/8.0;
	c2  = -1.0/24.0;
	dth = DT/DH;

	#pragma omp parallel for private (i, j, k)
	for (i = 4; i <= nxt+3; i++)
		for (j = 8; j <= nyt-1; j++)
			for (k = align; k <= nzt+align-1; k++)
			{
						register int dcrj, d_1, d_2, d_3;
						register int pos;
						register int pos_ip1, pos_ip2;
						register int pos_im1, pos_im2;
						register int pos_jm1, pos_jm2;
						register int pos_jp1, pos_jp2;
						register int pos_km1, pos_km2;
						register int pos_kp1, pos_kp2;
						register int pos_jk1, pos_ik1, pos_ij1;

						pos		= i*slice_1+j*yline_1+k;
						pos_km2 = pos-2;
						pos_km1 = pos-1;
						pos_kp1 = pos+1;
						pos_kp2 = pos+2;
						pos_jm2 = pos-yline_2;
						pos_jm1 = pos-yline_1;
						pos_jp1 = pos+yline_1;
						pos_jp2 = pos+yline_2;
						pos_im1 = pos-slice_1;
						pos_im2 = pos-slice_2;
						pos_ip1 = pos+slice_1;
						pos_ip2 = pos+slice_2;
						pos_jk1 = pos-yline_1-1;
						pos_ik1 = pos+slice_1-1;
						pos_ij1 = pos+slice_1-yline_1;

						dcrj = dcrjx[i]*dcrjy[j]*dcrjz[k];

						d_1 = 0.25*((d1[pos]+d1[pos_jm1])+(d1[pos_km1]+d1[pos_jk1]));
						d_2 = 0.25*((d1[pos]+d1[pos_ip1])+(d1[pos_km1]+d1[pos_ik1]));
						d_3 = 0.25*((d1[pos]+d1[pos_ip1])+(d1[pos_jm1]+d1[pos_ij1]));

						u1[pos] = (u1[pos] + (dth/d_1)*(c1*(xx[pos]-xx[pos_im1]) + c2*(xx[pos_ip1]-xx[pos_im2])
													  + c1*(xy[pos]-xy[pos_jm1]) + c2*(xy[pos_jp1]-xy[pos_jm2])
													  + c1*(xz[pos]-xz[pos_km1]) + c2*(xz[pos_kp1]-xz[pos_km2]))) * dcrj;

						v1[pos] = (v1[pos] + (dth/d_2)*(c1*(xy[pos_ip1]-xy[pos]) + c2*(xy[pos_ip2]-xy[pos_im1])
													  + c1*(yy[pos_jp1]-yy[pos]) + c2*(yy[pos_jp2]-yy[pos_jm1])
													  + c1*(yz[pos]-yz[pos_km1]) + c2*(yz[pos_kp1]-yz[pos_km2]))) * dcrj;

						w1[pos] = (w1[pos] + (dth/d_3)*(c1*(xz[pos_ip1]-xz[pos]) + c2*(xz[pos_ip2]-xz[pos_im1])
													  + c1*(yz[pos]-yz[pos_jm1]) + c2*(yz[pos_jp1]-yz[pos_jm2])
													  + c1*(zz[pos_kp1]-zz[pos]) + c2*(zz[pos_kp2]-zz[pos_km1]))) * dcrj;

			}
	return;
}

void verification(int nxt, int nyt, int nzt, float *p1, float *p2)
{
	int i, j, k;
	float diff = 0.0, m_p1 = 0.0, m_p2 = 0.0;
	int pos;
	for (i = 4; i <= nxt+3; i++)
		for (j = 8; j <= nyt-1; j++)
			for (k = align; k <= nzt+align-1; k++)
			{
				pos = i*(nyt+8)*(nzt+2*align) + j*(nzt+2*align) + k;
				m_p1 += p1[pos]*p1[pos];
				m_p2 += p2[pos]*p2[pos];
				diff += (p1[pos]-p2[pos])*(p1[pos]-p2[pos]);
			}
	printf ("%f\n", diff/(m_p1*m_p2));
}
int main()
{
	int nxt, nyt, nzt;
	int i, j, k;
	float *u1, *v1, *w1, *u2, *v2, *w2, *xx, *yy, *zz, *xy, *xz, *yz, *d1, *dcrjx, *dcrjy, *dcrjz;

	nxt = 512;
	nyt = 512;
	nzt = 512;

	u1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	v1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	w1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	u2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	v2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	w2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	xx = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yy = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	zz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xy = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	d1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	dcrjx = (float*) malloc ((nxt+8)*sizeof(float));
	dcrjy = (float*) malloc ((nyt+8)*sizeof(float));
	dcrjz = (float*) malloc ((nzt+2*align)*sizeof(float));

	if (u1 == NULL) printf ("Allocate u1 failed!\n");
	if (v1 == NULL) printf ("Allocate v1 failed!\n");
	if (w1 == NULL) printf ("Allocate w1 failed!\n");
	if (u2 == NULL) printf ("Allocate u2 failed!\n");
	if (v2 == NULL) printf ("Allocate v2 failed!\n");
	if (w2 == NULL) printf ("Allocate w2 failed!\n");
	if (xx == NULL) printf ("Allocate xx failed!\n");
	if (yy == NULL) printf ("Allocate yy failed!\n");
	if (zz == NULL) printf ("Allocate zz failed!\n");
	if (xy == NULL) printf ("Allocate xy failed!\n");
	if (xz == NULL) printf ("Allocate xz failed!\n");
	if (yz == NULL) printf ("Allocate uz failed!\n");
	if (d1 == NULL) printf ("Allocate d1 failed!\n");
	if (dcrjx == NULL) printf ("Allocate dcrjx failed!\n");
	if (dcrjy == NULL) printf ("Allocate dcrjy failed!\n");
	if (dcrjz == NULL) printf ("Allocate dcrjz failed!\n");

	for (i = 0; i < nxt+8; i++) dcrjx[i] = i;
	for (j = 0; j < nyt+8; j++) dcrjy[j] = j;
	for (k = align; k < nzt+align; k++) dcrjz[k] = k;

	for (i = 0; i < nxt+8; i++)
	  for (j = 0; j < nyt+8; j++)
		for (k = align; k < nzt+align; k++)
		{
			int pos = i*(nyt+8)*(nzt+2*align) + j*(nzt+2*align) + k;
			d1[pos] = pos;
			xx[pos] = pos;
			yy[pos] = pos;
			zz[pos] = pos;
			xy[pos] = pos;
			xz[pos] = pos;
			yz[pos] = pos;
		}

	struct timespec s1, e1, s2, e2;
	double t1, t2;

	clock_gettime(CLOCK_REALTIME, &s1);
	dvelcx_omp(1.0, 1.0, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1);
	clock_gettime(CLOCK_REALTIME, &e1);

	t1 = (e1.tv_sec - s1.tv_sec);
	t1 += (e1.tv_nsec - s1.tv_nsec) / 1000000000.0;
	printf ("%f\n", t1);

	clock_gettime(CLOCK_REALTIME, &s2);
	dvelcx_omp_2(1.0, 1.0, nxt, nyt, nzt, u2, v2, w2, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1);
	clock_gettime(CLOCK_REALTIME, &e2);

	t2 = (e2.tv_sec - s2.tv_sec);
	t2 += (e2.tv_nsec - s2.tv_nsec) / 1000000000.0;
	printf ("%f\n", t2);

	verification(nxt, nyt, nzt, u1, u2);
	verification(nxt, nyt, nzt, v1, v2);
	verification(nxt, nyt, nzt, w1, w2);

	return 0;
}
