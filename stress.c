#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#define iblock 32
#define jblock 32
#define kblock 32
#define align 0
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

void dstrqc(float DH,  float DT,
		int nxt, int nyt, int nzt, int NX,
		float *ptr_vx1, float *ptr_vx2,
		float *ptr_u1, float *ptr_v1, float *ptr_w1,
		float *ptr_xx, float *ptr_yy, float *ptr_zz, float *ptr_xy, float *ptr_xz, float *ptr_yz,
		float *ptr_r1, float *ptr_r2, float *ptr_r3, float *ptr_r4, float *ptr_r5, float *ptr_r6,
		float *ptr_lam,float *ptr_mu, float *ptr_qp, float *ptr_qs, float *ptr_lam_mu,
		float *ptr_dcrjx, float *ptr_dcrjy, float *ptr_dcrjz,
		int rankx, int ranky, int e_i, int e_j, int i_s, int j_s, int k_s)
{
	const float c1 = 9.0/8.0, c2 = -1.0/24.0;
	const float dt1 = 1.0/DT;
	const float dh1 = 1.0/DH;
	const float dth = DT/DH;
	const int   slice_1 = (nyt+8)*(nzt+2*align);
	const int 	slice_2 = (nyt+8)*(nzt+2*align)*2;
	const int	yline_1 = nzt+2*align;
	const int	yline_2 = (nzt+2*align)*2;
	const int i_e = MIN(i_s+iblock-1, e_i);
	const int j_e = MIN(j_s+jblock-1, e_j);
	const int k_e = MIN(k_s+kblock-1, nzt+align-1);

	int ii, jj, kk;


	for (ii = i_s; ii <= i_e; ii++)
	{
		float dcrjx = ptr_dcrjx[ii];
		float *vx1 = &ptr_vx1[ii*slice_1+j_s*yline_1];
		float *vx2 = &ptr_vx2[ii*slice_1+j_s*yline_1];
		float *u1 = &ptr_u1[ii*slice_1+j_s*yline_1];
		float *v1 = &ptr_v1[ii*slice_1+j_s*yline_1];
		float *w1 = &ptr_w1[ii*slice_1+j_s*yline_1];
		float *xx = &ptr_xx[ii*slice_1+j_s*yline_1];
		float *yy = &ptr_yy[ii*slice_1+j_s*yline_1];
		float *zz = &ptr_zz[ii*slice_1+j_s*yline_1];
		float *xy = &ptr_xy[ii*slice_1+j_s*yline_1];
		float *xz = &ptr_xz[ii*slice_1+j_s*yline_1];
		float *yz = &ptr_yz[ii*slice_1+j_s*yline_1];
		float *r1 = &ptr_r1[ii*slice_1+j_s*yline_1];
		float *r2 = &ptr_r2[ii*slice_1+j_s*yline_1];
		float *r3 = &ptr_r3[ii*slice_1+j_s*yline_1];
		float *r4 = &ptr_r4[ii*slice_1+j_s*yline_1];
		float *r5 = &ptr_r5[ii*slice_1+j_s*yline_1];
		float *r6 = &ptr_r6[ii*slice_1+j_s*yline_1];
		float *lam = &ptr_lam[ii*slice_1+j_s*yline_1];
		float *mu = &ptr_mu[ii*slice_1+j_s*yline_1];
		float *qp = &ptr_qp[ii*slice_1+j_s*yline_1];
		float *qs = &ptr_qs[ii*slice_1+j_s*yline_1];
		for (jj = j_s; jj <= j_e; jj++)
		{
			float dcrjy = ptr_dcrjy[jj];
			#pragma vector always
			#pragma simd
			for (kk = k_s; kk <= k_e; kk++)
			{

				register float f_vx1 = vx1[kk];
				register float f_vx2 = vx2[kk];
				register float dcrj = dcrjx*dcrjy*ptr_dcrjz[kk];

				register float xl = 8.0/(lam[kk] + lam[kk+slice_1] + lam[kk-yline_1] + lam[kk+slice_1-yline_1]
							        + lam[kk-1] + lam[kk+slice_1-1] + lam[kk-yline_1-1] + lam[kk+slice_1-yline_1-1]);
				register float xm = 16.0/(mu[kk] + mu[kk+slice_1] + mu[kk-yline_1] + mu[kk+slice_1-yline_1]
							         + mu[kk-1] + mu[kk+slice_1-1] + mu[kk-yline_1-1] + mu[kk+slice_1-yline_1-1]);

				register float xmu1 = 2.0/(mu[kk] + mu[kk-1]);
				register float xmu2 = 2.0/(mu[kk] + mu[kk-yline_1]);
				register float xmu3 = 2.0/(mu[kk] + mu[kk+slice_1]);

				xl = xl + xm;

				register float qpa = 0.0625*(qp[kk] + qp[kk+slice_1] + qp[kk-yline_1] + qp[kk+slice_1-yline_1]
					                    + qp[kk-1] + qp[kk+slice_1-1] + qp[kk-yline_1-1] + qp[kk+slice_1-yline_1-1]);
				register float h 	= 0.0625*(qs[kk] + qs[kk+slice_1] + qs[kk-yline_1] + qs[kk+slice_1-yline_1]
					                    + qs[kk-1] + qs[kk+slice_1-1] + qs[kk-yline_1-1] + qs[kk+slice_1-yline_1-1]);

				register float h1 = 0.250*(qs[kk] + qs[kk-1]);
				register float h2 = 0.250*(qs[kk] + qs[kk-yline_1]);
				register float h3 = 0.250*(qs[kk] + qs[kk+slice_1]);
				register float vx;

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

				register float vs1, vs2, vs3;
				register int g_i;

				if (kk == nzt+align-1)
				{
					u1[kk+1] = u1[kk] - (w1[kk] - w1[kk-slice_1]);
					v1[kk+1] = v1[kk] - (w1[kk+yline_1] - w1[kk]);

					g_i  = nxt*rankx + ii - 3; /* uncertainty */
					if (g_i < NX)
						vs1	= u1[kk+slice_1] - (w1[kk+slice_1] - w1[kk]);
					else
						vs1 = 0.0;

					g_i = nyt*ranky + ii - 3; /* uncertainty */
					if(g_i > 1)
						vs2	= v1[kk-yline_1] - (w1[kk] - w1[kk-yline_1]);
					else
						vs2 = 0.0;

					w1[kk+1]	= w1[kk-1] - ptr_lam_mu[ii*(nyt+8)+jj]*((vs1 - u1[kk+1])
							+ (u1[kk+slice_1] - u1[kk])
							+ (v1[kk+1] - vs2)
							+ (v1[kk] - v1[kk-yline_1]));
				}

				if (kk == nzt+align-2)
				{
					u1[kk+2] = u1[kk+1] - (w1[kk+1] - w1[kk-slice_1+1]);
					v1[kk+2] = v1[kk+1] - (w1[kk+yline_1+1] - w1[kk+1]);
				}

				vs1 = c1*(u1[kk+slice_1] - u1[kk]) + c2*(u1[kk+slice_2] - u1[kk-slice_1]);
				vs2 = c1*(v1[kk] - v1[kk-yline_1]) + c2*(v1[kk+yline_1] - v1[kk-yline_2]);
				vs3 = c1*(w1[kk] - w1[kk-1]) + c2*(w1[kk+1] - w1[kk-2]);

				register float a1, tmp;
				tmp = xl*(vs1+vs2+vs3);
				a1  = qpa*(vs1+vs2+vs3);
				tmp = tmp+DT*a1;

				register float r;

				r = r1[kk];
				xx[kk] = (xx[kk] + tmp - xm*(vs2+vs3) + vx*r)*dcrj;
				r1[kk] = f_vx2*r - h*(vs2+vs3) + a1;

				r = r2[kk];
				yy[kk] = (yy[kk] + tmp - xm*(vs1+vs3) + vx*r)*dcrj;
				r2[kk] = f_vx2*r - h*(vs1+vs3) + a1;

				r = r3[kk];
				zz[kk] = (zz[kk] + tmp - xm*(vs1+vs2) + vx*r)*dcrj;
				r3[kk] = f_vx2*r - h*(vs1+vs2) + a1;

				vs1 = c1*(u1[kk+yline_1] - u1[kk]) + c2*(u1[kk+yline_2] - u1[kk-yline_1]);
				vs2 = c1*(v1[kk] - v1[kk-slice_1]) + c2*(v1[kk+slice_1] - v1[kk-slice_2]);

				r = r4[kk];
				xy[kk] = (xy[kk] + xmu1*(vs1+vs2) + vx*r)*dcrj;
				r4[kk] = f_vx2*r + h1*(vs1+vs2);

				if(kk == nzt+align-1)
				{
					zz[kk+1] = -zz[kk];
					xz[kk]   = 0.0;
					yz[kk]   = 0.0;
				}
				else
				{
					vs1 = c1*(u1[kk+1] - u1[kk]) + c2*(u1[kk+2] - u1[kk-1]);
					vs2 = c1*(w1[kk] - w1[kk-slice_1]) + c2*(w1[kk+slice_1] - w1[kk-slice_2]);

					r = r5[kk];
					xz[kk] = (xz[kk] + xmu2*(vs1+vs2) + vx*r)*dcrj;
					r5[kk] = f_vx2*r + h2*(vs1+vs2);

					vs1 = c1*(v1[kk+1] - v1[kk]) + c2*(v1[kk+2] - v1[kk-1]);
					vs2 = c1*(w1[kk+yline_1] - w1[kk]) + c2*(w1[kk+yline_2] - w1[kk-yline_1]);

					r = r6[kk];
					yz[kk] = (yz[kk] + xmu3*(vs1+vs2) + vx*r)*dcrj;
					r6[kk] = f_vx2*r + h3*(vs1+vs2);

					if(k == nzt+align-2)
					{
						zz[kk+3] = -zz[kk];
						xz[kk+2] = -xz[kk];
						yz[kk+2] = -yz[kk];
					}
					else
						if(kk == nzt+align-3)
						{
							xz[kk+4] = -xz[kk];
							yz[kk+4] = -yz[kk];
						}
				}
			}
		}
	}
	return;
}
void dstrqc_omp(float DH,  float DT,
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
	int index = 0;
	int num_blocks, num_i_blocks, num_j_blocks, num_k_blocks;
	int *blocking;

	num_i_blocks = ceil((double)nxt/iblock);
	num_j_blocks = ceil((double)nyt/jblock);
	num_k_blocks = ceil((double)nzt/kblock);
	num_blocks = num_i_blocks * num_j_blocks * num_k_blocks;


	blocking = (int*)malloc(3*num_blocks*sizeof(int));
	if (blocking == NULL) printf ("Allocate blocking failed!\n");

	for (i = s_i; i <= e_i; i += iblock)
		for (j = s_j; j <= e_j; j += jblock)
			for (k = align; k <= nzt+align-1; k += kblock)
			{
				blocking[index++] = i;
				blocking[index++] = j;
				blocking[index++] = k;
			}

	#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < num_blocks; i++)
	{
		int i_s = blocking[3*i];
		int j_s = blocking[3*i+1];
		int k_s = blocking[3*i+2];
		dstrqc(DH, DT, nxt, nyt, nzt, NX, vx1, vx2, u1, v1, w1, xx, yy, zz, xy, xz, yz, r1, r2, r3, r4, r5, r6, lam, mu, qp, qs, lam_mu, dcrjx, dcrjy, dcrjz,
			rankx, ranky, e_i, e_j, i_s, j_s, k_s);
	}

	free(blocking);
	return;
}

void dstrqc_omp_2(float DH,  float DT,
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
	int slice_1,  slice_2,  yline_1,  yline_2;
	int g_i;
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


	#pragma omp parallel for private(i, j, k, g_i, f_vx1, f_vx2, xl, xm, xmu1, xmu2, xmu3, qpa, h, h1, h2, h3, vs1, vs2, vs3, a1, tmp, dcrj, vx)
	for (i = s_i; i <= e_i; i++)
		for (j = s_j; j <= e_j; j++)
			for (k = align; k <= align+nzt-1; k++)
						{
							register int pos;
							register int pos_ip1, pos_ip2;
							register int pos_im1, pos_im2;
							register int pos_km1, pos_km2;
							register int pos_kp1, pos_kp2;
							register int pos_jm1, pos_jm2;
							register int pos_jp1, pos_jp2;
							register int pos_ik1, pos_jk1, pos_ij1, pos_ijk1;

							pos = i*slice_1+j*yline_1+k;
							pos_km2 = pos-2;
							pos_km1 = pos-1;
							pos_kp1 = pos+1;
							pos_kp2 = pos+2;
							pos_jm1 = pos-yline_1;
							pos_jm2 = pos-yline_2;
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
							dcrj = dcrjx[i]*dcrjy[j]*dcrjz[k];

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

			                if (k == nzt+align-1)
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

			                	w1[pos_kp1]	= w1[pos_km1] - lam_mu[i*(nyt+8)+j]*((vs1 - u1[pos_kp1])
			                												   + (u1[pos_ip1] - u1[pos])
																			   + (v1[pos_kp1] - vs2)
																			   + (v1[pos] - v1[pos_jm1]));
			                }

			                if (k == nzt+align-2)
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
	int NX;
	int i, j, k;
	float *u1, *v1, *w1, *xx, *yy, *zz, *xy, *xz, *yz, *dcrjx, *dcrjy, *dcrjz;
	float *vx1, *vx2, *r1, *r2, *r3, *r4, *r5, *r6, *lam, *mu, *qp, *qs, *lam_mu;

	nxt = 512;
	nyt = 512;
	nzt = 128;
	NX = 512;

	u1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	v1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	w1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	xx_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yy_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	zz_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xy_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xz_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yz_1 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

	xx_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yy_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	zz_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xy_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	xz_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));
	yz_2 = (float*) malloc ((nxt+8)*(nyt+8)*(nzt+2*align)*sizeof(float));

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
	if (xx_1 == NULL) printf ("Allocate xx_1 failed!\n");
	if (yy_1 == NULL) printf ("Allocate yy_1 failed!\n");
	if (zz_1 == NULL) printf ("Allocate zz_1 failed!\n");
	if (xy_1 == NULL) printf ("Allocate xy_1 failed!\n");
	if (xz_1 == NULL) printf ("Allocate xz_1 failed!\n");
	if (yz_1 == NULL) printf ("Allocate zz_1 failed!\n");
	if (xx_2 == NULL) printf ("Allocate xx_2 failed!\n");
	if (yy_2 == NULL) printf ("Allocate yy_2 failed!\n");
	if (zz_2 == NULL) printf ("Allocate zz_2 failed!\n");
	if (xy_2 == NULL) printf ("Allocate xy_2 failed!\n");
	if (xz_2 == NULL) printf ("Allocate xz_2 failed!\n");
	if (yz_2 == NULL) printf ("Allocate zz_2 failed!\n");
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

	for (i = 0; i < nxt+8; i++) dcrjx[i] = i;
	for (j = 0; j < nyt+8; j++) dcrjy[j] = j;
	for (k = align; k < nzt+align; k++) dcrjz[k] = k;

	for (i = 0; i < nxt+8; i++)
	  for (j = 0; j < nyt+8; j++)
		for (k = align; k < nzt+align; k++)
		{
			int pos = i*(nyt+8)*(nzt+2*align) + j*(nzt+2*align) + k;
			u1[pos] = pos;
			v1[pos] = pos;
			w1[pos] = pos;
			xx_1[pos] = pos;
			yy_1[pos] = pos;
			zz_1[pos] = pos;
			xy_1[pos] = pos;
			xz_1[pos] = pos;
			yz_1[pos] = pos;
			xx_2[pos] = pos;
			yy_2[pos] = pos;
			zz_2[pos] = pos;
			xy_2[pos] = pos;
			xz_2[pos] = pos;
			yz_2[pos] = pos;
			lam[pos] = pos;
			mu[pos] = pos;
			qp[pos] = pos;
			qs[pos] = pos;
		}

	for (i = 0; i < nxt+8; i++)
		for (j = 0; j < nyt+8; j++)
		{
			int pos = i*(nyt+8) + j;
			lam_mu[pos] = pos;
		}

	struct timespec s3, e3, s4, e4;
	double t3, t4;

	clock_gettime(CLOCK_REALTIME, &s3);
	dstrqc_omp(1.0, 1.0, nxt, nyt, nzt, NX, vx1, vx2, u1, v1, w1, xx_1, yy_1, zz_1, xy_1, xz_1, yz_1, r1, r2, r3, r4, r5, r6, lam, mu, qp, qs, lam_mu, dcrjx, dcrjy,dcrjz, 0, 0, 4, nxt+3, 4, nyt+3);
	clock_gettime(CLOCK_REALTIME, &e3);

	clock_gettime(CLOCK_REALTIME, &s3);
	dstrqc_omp_2(1.0, 1.0, nxt, nyt, nzt, NX, vx1, vx2, u1, v1, w1, xx_2, yy_2, zz_2, xy_2, xz_2, yz_2, r1, r2, r3, r4, r5, r6, lam, mu, qp, qs, lam_mu, dcrjx, dcrjy,dcrjz, 0, 0, 4, nxt+3, 4, nyt+3);
	clock_gettime(CLOCK_REALTIME, &e3);

	t3 = (e3.tv_sec - s3.tv_sec);
	t3 += (e3.tv_nsec - s3.tv_nsec) / 1000000000.0;
	printf ("%f\n", t3);

	t4 = (e4.tv_sec - s4.tv_sec);
	t4 += (e4.tv_nsec - s4.tv_nsec) / 1000000000.0;
	printf ("%f\n", t4);

	verfication(nxt, nyt, nzt, xx_1, xx_2);

	return 0;
}
