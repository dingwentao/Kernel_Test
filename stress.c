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

void dstrqc(float DH,  float DT,
			int nxt, int nyt, int nzt, int NX,
			float *vx1, float *vx2,
			float *u1, float *v1, float *w1,
			float *xx, float *yy, float *zz, float *xy, float *xz, float *yz,
			float* r1, float* r2, float* r3, float* r4, float* r5, float* r6,
			float* lam,float* mu, float* qp, float* qs, float* lam_mu,
			float* dcrjx, float* dcrjy, float* dcrjz,
			int rankx, int ranky, int s_i, int e_i, int s_j, int e_j)
{
	int i, j, k;
	int ii, jj, kk;
	int g_i;
	int slice_1,  slice_2,  yline_1,  yline_2;
	float dt1, dh1, dth, c1, c2;
	float f_vx1, f_vx2;
	float xl, xm;
	float xmu1, xmu2, xmu3;
	float qpa, h;
	float h1, h2, h3;
	float vs1, vs2, vs3;
	float a1, tmp, dcrj, vx;

	slice_1  = (nyt+8)*(nzt+2*align);
	slice_2  = (nyt+8)*(nzt+2*align)*2;
	yline_1  = nzt+2*align;
	yline_2  = (nzt+2*align)*2;

	dt1 = 1.0/DT;
	dh1 = 1.0/DH;
	dth = DT/DH;
	c1  = 9.0/8.0;
	c2  = -1.0/24.0;


	#pragma omp parallel for private(i, j, k, ii, jj, kk, g_i, f_vx1, f_vx2, xl, xm, xmu1, xmu2, xmu3, qpa, h, h1, h2, h3, vs1, vs2, vs3, a1, tmp, dcrj, vx)
	for (i = s_i; i <= e_i; i += iblock)
		for (j = s_j; j <= e_j; j += jblock)
			for (k = align; k <= align+nzt-1; k += kblock)
				for (ii = i; ii <= MIN(i+iblock, e_i); ii++)
					for (jj = j; jj <= MIN(j+jblock, e_j); jj++)
						#pragma ivdep
						for (kk = k; kk <= MIN(k+kblock, nzt+align-1); kk++)
						{
							register int pos;
							register int pos_ip1, pos_ip2;
							register int pos_im1, pos_im2;
							register int pos_km1, pos_km2;
							register int pos_kp1, pos_kp2;
							register int pos_jm1, pos_jm2;
							register int pos_jp1, pos_jp2;
							register int pos_ik1, pos_jk1, pos_ij1, pos_ijk1;

							pos = ii*slice_1+jj*yline_1+kk;
							pos_km1 = pos-1;
							pos_km2 = pos-2;
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
							pos_ijk1 = pos+slice_1-yline_1-1;

							f_vx1 = vx1[pos];
							f_vx2 = vx2[pos];
							dcrj = dcrjx[ii]*dcrjy[jj]*dcrjz[kk];

							xl = 8.0/(lam[pos] + lam[pos_ip1] + lam[pos_jm1] + lam[pos_ij1]
							        + lam[pos_km1] + lam[pos_ik1] + lam[pos_jk1] + lam[pos_ijk1]);
							xm = 16.0/(mu[pos] + mu[pos_ip1] + mu[pos_jm1] + mu[pos_ij1]
							         + mu[pos_km1] + mu[pos_ik1] + mu[pos_jk1] + mu[pos_ijk1]);

							xmu1 = 2.0/(mu[pos] + mu[pos_km1]);
					        xmu2 = 2.0/(mu[pos] + mu[pos_jm1]);
					        xmu3 = 2.0/(mu[pos] + mu[pos_ip1]);

					        xl = xl + xm;

					        qpa = 0.0625*(qp[pos] + qp[pos_ip1] + qp[pos_jm1] + qp[pos_ij1]
					                    + qp[pos_km1] + qp[pos_ik1] + qp[pos_jk1] + qp[pos_ijk1]);
					        h 	= 0.0625*(qs[pos] + qs[pos_ip1] + qs[pos_jm1] + qs[pos_ij1]
					                    + qs[pos_km1] + qs[pos_ik1] + qs[pos_jk1] + qs[pos_ijk1]);

					        h1 = 0.250*(qs[pos] + qs[pos_km1]);
					        h2 = 0.250*(qs[pos] + qs[pos_jm1]);
					        h3 = 0.250*(qs[pos] + qs[pos_ip1]);

			                h     = -xm*h*dh1;
			                h1    = -xmu1*h1*dh1;
			                h2    = -xmu2*h2*dh1;
			                h3    = -xmu3*h3*dh1;
			                qpa   = -qpa*xl*dh1;
			                xm    = xm*dth;
			                xmu1  = xmu1*dth;
			                xmu2  = xmu2*dth;
			                xmu3  = xmu3*dth;
			                xl    = xl*dth;
			                f_vx2 = f_vx2*f_vx1;
			                h     = h*f_vx1;
			                h1    = h1*f_vx1;
			                h2    = h2*f_vx1;
			                h3    = h3*f_vx1;
			                qpa   = qpa*f_vx1;

			                xm    = xm+DT*h;
			                xmu1  = xmu1+DT*h1;
			                xmu2  = xmu2+DT*h2;
			                xmu3  = xmu3+DT*h3;
			                vx    = DT*(1+f_vx2);

			                if (kk == nzt+align-1)
			                {
			                	u1[pos_kp1] = u1[pos] - (w1[pos] - w1[pos_im1]);
			                	v1[pos_kp1] = v1[pos] - (w1[pos_jp1] - w1[pos]);

			                	g_i  = nxt*rankx + i - 3; /* uncertainty */
								if (g_i < NX)
			                		vs1	= u1[pos_ip1] - (w1[pos_ip1] - w1[pos]);
			                	else
			                		vs1 = 0.0;

			                	g_i = nyt*ranky + i - 3; /* uncertainty */
			                	if(g_i > 1)
			                		vs2	= v1[pos_jm1] - (w1[pos] - w1[pos_jm1]);
			                	else
			                		vs2 = 0.0;

			                	w1[pos_kp1]	= w1[pos_km1] - lam_mu[ii*(nyt+8)+jj]*((vs1 - u1[pos_kp1])
			                												   + (u1[pos_ip1] - u1[pos])
																			   + (v1[pos_kp1] - vs2)
																			   + (v1[pos] - v1[pos_jm1]));
			                }

			                if (kk == nzt+align-2)
			                {
			                	 u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1] - w1[pos_im1+1]);
								 v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
			                }

			                vs1 = c1*(u1[pos_ip1] - u1[pos]) + c2*(u1[pos_ip2] - u1[pos_im1]);
			                vs2 = c1*(v1[pos] - v1[pos_jm1]) + c2*(v1[pos_jp1] - v1[pos_jm2]);
			                vs3 = c1*(w1[pos] - w1[pos_km1]) + c2*(w1[pos_kp1] - w1[pos_km2]);

			                tmp = xl*(vs1+vs2+vs3);
			                a1  = qpa*(vs1+vs2+vs3);
			                tmp = tmp+DT*a1;

			                register float r;

							r = r1[pos];
			                xx[pos] = (xx[pos] + tmp - xm*(vs2+vs3) + vx*r)*dcrj;
			                r1[pos] = f_vx2*r - h*(vs2+vs3) + a1;

			                r = r2[pos];
			                yy[pos] = (yy[pos] + tmp - xm*(vs1+vs3) + vx*r)*dcrj;
			                r2[pos] = f_vx2*r - h*(vs1+vs3) + a1;

			                r = r3[pos];
			                zz[pos] = (zz[pos] + tmp - xm*(vs1+vs2) + vx*r)*dcrj;
			                r3[pos] = f_vx2*r - h*(vs1+vs2) + a1;

			                vs1 = c1*(u1[pos_jp1] - u1[pos]) + c2*(u1[pos_jp2] - u1[pos_jm1]);
			                vs2 = c1*(v1[pos] - v1[pos_im1]) + c2*(v1[pos_ip1] - v1[pos_im2]);

			                r = r4[pos];
			                xy[pos] = (xy[pos] + xmu1*(vs1+vs2) + vx*r)*dcrj;
			                r4[pos] = f_vx2*r + h1*(vs1+vs2);

			                if(k == nzt+align-1)
			                {
			                	zz[pos+1] = -zz[pos];
			                	xz[pos]   = 0.0;
			                	yz[pos]   = 0.0;
			                }
			                else
			                {
			                	vs1 = c1*(u1[pos_kp1] - u1[pos]) + c2*(u1[pos_kp2] - u1[pos_km1]);
			                	vs2 = c1*(w1[pos] - w1[pos_im1]) + c2*(w1[pos_ip1] - w1[pos_im2]);

			                	r = r5[pos];
			                	xz[pos] = (xz[pos] + xmu2*(vs1+vs2) + vx*r)*dcrj;
			                	r5[pos] = f_vx2*r + h2*(vs1+vs2);

			                	vs1 = c1*(v1[pos_kp1] - v1[pos]) + c2*(v1[pos_kp2] - v1[pos_km1]);
			                	vs2 = c1*(w1[pos_jp1] - w1[pos]) + c2*(w1[pos_jp2] - w1[pos_jm1]);

			                	r = r6[pos];
			                	yz[pos] = (yz[pos] + xmu3*(vs1+vs2) + vx*r)*dcrj;
			                	r6[pos] = f_vx2*r + h3*(vs1+vs2);

			                	if(k == nzt+align-2)
			                	{
			                		zz[pos+3] = -zz[pos];
			                		xz[pos+2] = -xz[pos];
			                		yz[pos+2] = -yz[pos];
			                	}
			                	else
			                		if(k == nzt+align-3)
			                		{
			                			xz[pos+4] = -xz[pos];
			                			yz[pos+4] = -yz[pos];
			                		}
			                }
						}

	return;
}

int main()
{
	int nxt, nyt, nzt;
	int NX;
	int i, j, k;
	float *u1, *v1, *w1, *xx, *yy, *zz, *xy, *xz, *yz, *dcrjx, *dcrjy, *dcrjz;
	float *vx1, *vx2, *r1, *r2, *r3, *r4, *r5, *r6, *lam, *mu, *qp, *qs, *lam_mu;

	nxt = 2048;
	nyt = 2048;
	nzt = 128;
	NX = 2048;

	u1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	v1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	w1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	xx = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yy = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	zz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xy = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yz = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	dcrjx = (float*) malloc ((nxt+8)*sizeof(float));
	dcrjy = (float*) malloc ((nyt+8)*sizeof(float));
	dcrjz = (float*) malloc ((nzt+2*align)*sizeof(float));

	vx1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	vx2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	r1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	r2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	r3 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	r4 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	r5 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	r6 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	lam = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	mu = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	qp = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	qs = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	lam_mu = (float*) malloc ((nxt+8)*(nyt+8)*sizeof(float));

	if (u1 == NULL) printf ("Allocate u1 failed!\n");
	if (v1 == NULL) printf ("Allocate v1 failed!\n");
	if (w1 == NULL) printf ("Allocate w1 failed!\n");
	if (xx == NULL) printf ("Allocate xx failed!\n");
	if (yy == NULL) printf ("Allocate yy failed!\n");
	if (zz == NULL) printf ("Allocate zz failed!\n");
	if (xy == NULL) printf ("Allocate xy failed!\n");
	if (xz == NULL) printf ("Allocate xz failed!\n");
	if (yz == NULL) printf ("Allocate uz failed!\n");
	if (dcrjx == NULL) printf ("Allocate dcrjx failed!\n");
	if (dcrjy == NULL) printf ("Allocate dcrjy failed!\n");
	if (dcrjz == NULL) printf ("Allocate dcrjz failed!\n");
	if (vx1 == NULL) printf ("Allocate vx1 failed!\n");
	if (vx2 == NULL) printf ("Allocate vx2 failed!\n");
	if (r1 == NULL) printf ("Allocate r1 failed!\n");
	if (r2 == NULL) printf ("Allocate r2 failed!\n");
	if (r3 == NULL) printf ("Allocate r3 failed!\n");
	if (r4 == NULL) printf ("Allocate r4 failed!\n");
	if (r5 == NULL) printf ("Allocate r5 failed!\n");
	if (r6 == NULL) printf ("Allocate r6 failed!\n");
	if (lam == NULL) printf ("Allocate lam failed!\n");
	if (mu == NULL) printf ("Allocate mu failed!\n");
	if (qp == NULL) printf ("Allocate qp failed!\n");
	if (qs == NULL) printf ("Allocate qs failed!\n");
	if (lam_mu == NULL) printf ("Allocate lam_mu failed!\n");

	struct timespec s3, e3;
	double t3;

	clock_gettime(CLOCK_REALTIME, &s3);
	dstrqc(1.0, 1.0, nxt, nyt, nzt, NX, vx1, vx2, u1, v1, w1, xx, yy, zz, xy, xz, yz, r1, r2, r3, r4, r5, r6, lam, mu, qp, qs, lam_mu, dcrjx, dcrjy,dcrjz, 0, 0, 4, nxt+3, 4, nyt+3);
	clock_gettime(CLOCK_REALTIME, &e3);

	t3 = (e3.tv_sec - s3.tv_sec);
	t3 += (e3.tv_nsec - s3.tv_nsec) / 1000000000.0;
	printf ("%f\n", t3);

	return 0;
}
