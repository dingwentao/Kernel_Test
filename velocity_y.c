#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#define align 0
#define iblock 32
#define jblock 32
#define kblock 32
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

void dvelcy(float DH, float DT,
			int nxt, int nyt, int nzt,
			float* u1, float* v1, float* w1,
			float* xx, float* yy, float* zz, float* xy, float* xz, float* yz,
            float* dcrjx, float* dcrjy, float* dcrjz, float *d1,
			float* s_u1, float* s_v1, float* s_w1,
			int s_j, int e_j)
{
	int i, j, k;
	int ii, jj, kk;
	float c1, c2;
	float dth, dcrj;
	float d_1, d_2, d_3;
	int   slice_1,  slice_2,  yline_1,  yline_2;
	int j2;

	slice_1  = (nyt+8)*(nzt+2*align);
	slice_2  = (nyt+8)*(nzt+2*align)*2;
	yline_1  = nzt+2*align;
	yline_2  = (nzt+2*align)*2;

	c1 = 9.0/8.0;
	c2  = -1.0/24.0;
	dth = DT/DH;


	#pragma omp parallel for private (i, j, k, ii, jj, kk, dcrj, d_1, d_2, d_3, j2)
	for (i = 4; i <= nxt+3; i += iblock)
		for (k = align; k <= nzt+align-1; k += kblock)
			for (ii = i; ii <= MIN(i+iblock, nxt+3); ii++)
				for (jj = s_j, j2 = 0; jj <= e_j; jj++, j2++)
					#pragma ivdep
					for (kk = k; kk <= MIN(k+kblock, nzt+align-1); kk++)
					{
						register int pos;
						register int pos2;
						register int pos_ip1, pos_ip2;
						register int pos_im1, pos_im2;
						register int pos_jm1, pos_jm2;
						register int pos_jp1, pos_jp2;
						register int pos_km1, pos_km2;
						register int pos_kp1, pos_kp2;
						register int pos_jk1, pos_ik1, pos_ij1;

						pos		= ii*slice_1+jj*yline_1+kk;
						pos2	= ii*4*yline_1+j2*yline_1+kk;
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

						dcrj = dcrjx[ii]*dcrjy[jj]*dcrjz[kk];

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

						s_u1[pos2] = u1[pos];
						s_v1[pos2] = v1[pos];
						s_w1[pos2] = w1[pos];
					}
	return;
}


int main()
{
	int nxt, nyt, nzt;
	int i, j, k;
	float *u1, *v1, *w1, *xx, *yy, *zz, *xy, *xz, *yz, *d1, *dcrjx, *dcrjy, *dcrjz, *s_u1, *s_v1, *s_w1;

	nxt = 2048;
	nyt = 2048;
	nzt = 128;

	u1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	v1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	w1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

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

	s_u1 = (float*) malloc ((nxt+8)*4*(nzt+2*align)*sizeof(float));
	s_v1 = (float*) malloc ((nxt+8)*4*(nzt+2*align)*sizeof(float));
	s_w1 = (float*) malloc ((nxt+8)*4*(nzt+2*align)*sizeof(float));

	if (u1 == NULL) printf ("Allocate u1 failed!\n");
	if (v1 == NULL) printf ("Allocate v1 failed!\n");
	if (w1 == NULL) printf ("Allocate w1 failed!\n");
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
	if (s_u1 == NULL) printf ("Allocate s_u1 failed!\n");
	if (s_v1 == NULL) printf ("Allocate s_v1 failed!\n");
	if (s_w1 == NULL) printf ("Allocate s_w1 failed!");

	struct timespec s2, e2;
	double t2;


	clock_gettime(CLOCK_REALTIME, &s2);
	dvelcy(1.0, 1.0, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, s_u1, s_v1, s_w1, 4, 7);
	dvelcy(1.0, 1.0, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, s_u1, s_v1, s_w1, nyt, nyt+3);
	clock_gettime(CLOCK_REALTIME, &e2);

	t2 = (e2.tv_sec - s2.tv_sec);
	t2 += (e2.tv_nsec - s2.tv_nsec) / 1000000000.0;
	printf ("%f\n", t2);

	return 0;
}
