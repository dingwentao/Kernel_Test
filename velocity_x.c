#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define align 0
#define iblock 16
#define jblock 16
#define kblock 16
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

void dvelcx(float DH, float DT,
			int nxt, int nyt, int nzt,
			float* u1, float* v1, float* w1,
			float* xx, float* yy, float* zz, float* xy, float* xz, float* yz,
			float* dcrjx, float* dcrjy, float* dcrjz, float* d1, int i_s, int j_s, int k_s)
{
	float c1, c2;
	float dth, dcrj;
	float d_1, d_2, d_3;
	int   slice_1,  slice_2,  yline_1,  yline_2;

	slice_1  = (nyt+8)*(nzt+2*align);
	slice_2  = (nyt+8)*(nzt+2*align)*2;
	yline_1  = nzt+2*align;
	yline_2  = (nzt+2*align)*2;

	c1 = 9.0/8.0;
	c2  = -1.0/24.0;
	dth = DT/DH;

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
	num_j_blocks = ceil((double)nyt/jblock);
	num_k_blocks = ceil((double)nzt/kblock);
	num_blocks = num_i_blocks * num_j_blocks * num_k_blocks;

	printf ("%d %d %d %d\n", num_i_blocks, num_j_blocks, num_k_blocks, num_blocks);
	blocking = (int*)malloc(3*num_blocks*sizeof(int));
	if (blocking == NULL) printf ("Allocate blocking failed!\n");

	for (k = align; k <= nzt+align-1; k += kblock)
		for (j = 8; j <= nyt-1; j += jblock)
			for (i = 4; i <= nxt+3; i += iblock)
			{
				blocking[index++] = i;
				blocking[index++] = j;
				blocking[index++] = k;
			}
	printf ("%d\n", index);

	#pragma omp parallel for schedule(dynamic) firstprivate (nxt, nyt, nzt, num_blocks) shared (DH, DT, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, blocking)
	for (i = 0; i < num_blocks; i++)
	{
		int i_s = blocking[3*i];
		int j_s = blocking[3*i+1];
		int k_s = blocking[3*i+2];
		printf ("%d %d %d %d\n", omp_get_thread_num(), i_s, j_s, k_s);
		dvelcx(DT, DH, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1, i_s, j_s, k_s);
	}

	free(blocking);

/*	#pragma omp parallel for private (i, j, k, dcrj, d_1, d_2, d_3)
	for (i = 4; i <= nxt+3; i++)
		for (j = 8; j <= nyt-1; j++)
			for (k = align; k <= nzt+align-1; k++)
			{
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
	*/
	return;
}

int main()
{
	int nxt, nyt, nzt;
	int i, j, k;
	float *u1, *v1, *w1, *xx, *yy, *zz, *xy, *xz, *yz, *d1, *dcrjx, *dcrjy, *dcrjz;

	nxt = 32;
	nyt = 32;
	nzt = 32;

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


	struct timespec s1, e1;
	double t1;

	clock_gettime(CLOCK_REALTIME, &s1);
	dvelcx_omp(1.0, 1.0, nxt, nyt, nzt, u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d1);
	clock_gettime(CLOCK_REALTIME, &e1);

	t1 = (e1.tv_sec - s1.tv_sec);
	t1 += (e1.tv_nsec - s1.tv_nsec) / 1000000000.0;
	printf ("%f\n", t1);

	return 0;
}
