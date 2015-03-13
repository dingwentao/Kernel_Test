#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#define align 0
#define iblock 32
#define jblock 32
#define kblock 32
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )


int main(int arg, char **argv)
{
	int nxt, nyt, nzt;
	int i, j, k;
	float *u1, *v1, *w1, *u2, *v2, *w2, *xx, *yy, *zz, *xy, *xz, *yz, *d1, *dcrjx, *dcrjy, *dcrjz;
	int index;
	int num_blocks, num_i_blocks, num_j_blocks, num_k_blocks;
	int *blocking;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

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

	num_i_blocks = ceil((double)nxt/iblock);
	num_j_blocks = ceil((double)(nyt-8)/jblock);
	num_k_blocks = ceil((double)nzt/kblock);
	num_blocks = num_i_blocks * num_j_blocks * num_k_blocks;

	blocking = (int*)malloc(3*num_blocks*sizeof(int));
	if (blocking == NULL) printf ("Allocate blocking failed!\n");

	index = 0;
	for (i = 4; i <= nxt+3; i += iblock)
		for (j = 8; j <= nyt-1; j += jblock)
			for (k = align; k <= nzt+align-1; k += kblock)
			{
				blocking[index++] = i;
				blocking[index++] = j;
				blocking[index++] = k;
			}

    MPI_Finalize();
    return 0;
}
