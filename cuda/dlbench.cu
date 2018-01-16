#include<fstream>
#include<stdio.h>
#include<cstdlib>
#include<sys/time.h>
#include<pthread.h>
#include<dlbench.h>

#define MAX_ERRORS 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#ifdef AOS
__global__ void grayscale(pixel *src_images, pixel *dst_images) {

  int tidx = threadIdx.x; // + blockDim.x * blockIdx.x;

  int sets = (blockIdx.x / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  tidx = (tidx * SPARSITY) + (blockIdx.x - SPARSITY * sets) + set_offset;

  float F0 = 0.02f; 
  float F1 = 0.30f; 

      for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      //      DATA_ITEM_TYPE c0 = (src_images[tidx + j].r / src_images[tidx + j].g + src_images[tidx + j].b) // / (F1 * src_images[tidx + j].b);
      DATA_ITEM_TYPE c0 = (src_images[tidx + j].r / src_images[tidx + j].g + (F0 + F1 * F1) * 
			   src_images[tidx + j].b) / (F1 * src_images[tidx + j].b);
      DATA_ITEM_TYPE c1 = F1 * src_images[tidx + j].b;
//#pragma unroll UNROLL
     for (int k = 0; k < ITERS; k++) {
       v0 = v0 + c0;
       v1 = v0 - c1;
     }
     dst_images[tidx + j].r = (src_images[tidx + j].r * v0 - src_images[tidx + j].g * F1 * v1);
     dst_images[tidx + j].g  = (src_images[tidx + j].g * F1 * (1.0f - v1) - src_images[tidx + j].r); 
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].b = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].x = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].a = v0; 
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].c  = v1; 
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].d = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].e = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].f = v0; 
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].h  = v1; 
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].j = v0; 
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].k  = v1; 
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].l = v0; 
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].m  = v1; 
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dst_images[tidx + j].n  = v1; 
#endif
#if defined MEM16 || MEM17 || MEM18
      dst_images[tidx + j].o  = v1; 
#endif
#if defined MEM17 || MEM18
      dst_images[tidx + j].p  = v1; 
#endif
#if defined MEM18
      dst_images[tidx + j].q  = v1; 
#endif
    }
}
#endif

#ifdef DA
__global__ void grayscale( DATA_ITEM_TYPE *r,  DATA_ITEM_TYPE *g, 
			   DATA_ITEM_TYPE *b,  DATA_ITEM_TYPE *x,  
			   DATA_ITEM_TYPE *a,  DATA_ITEM_TYPE *c, 
			   DATA_ITEM_TYPE *d,  DATA_ITEM_TYPE *e,  
			   DATA_ITEM_TYPE *f,  DATA_ITEM_TYPE *h, 
			   DATA_ITEM_TYPE *j,  DATA_ITEM_TYPE *k,  
			   DATA_ITEM_TYPE *l,  DATA_ITEM_TYPE *m, 
			   DATA_ITEM_TYPE *n,  DATA_ITEM_TYPE *o,  
			   DATA_ITEM_TYPE *p,  DATA_ITEM_TYPE *q, 
			   DATA_ITEM_TYPE *d_r, DATA_ITEM_TYPE *d_g,  
			   DATA_ITEM_TYPE *d_b,  DATA_ITEM_TYPE *d_x,
			   DATA_ITEM_TYPE *d_a,  DATA_ITEM_TYPE *d_c, 
			   DATA_ITEM_TYPE *d_d,  DATA_ITEM_TYPE *d_e,  
			   DATA_ITEM_TYPE *d_f,  DATA_ITEM_TYPE *d_h, 
			   DATA_ITEM_TYPE *d_j,  DATA_ITEM_TYPE *d_k,  
			   DATA_ITEM_TYPE *d_l,  DATA_ITEM_TYPE *d_m,
			   DATA_ITEM_TYPE *d_n,  DATA_ITEM_TYPE *d_o,  
			   DATA_ITEM_TYPE *d_p,  DATA_ITEM_TYPE *d_q) {

  int tidx = threadIdx.x; 

  int sets = (blockIdx.x / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  tidx = (tidx * SPARSITY) + (blockIdx.x - SPARSITY * sets) + set_offset;

  //  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  //  tidx = tidx * CF;  // coarsening factor
  //  THREADS/CF
  float F0 = 0.02f; 
  float F1 = 0.30f; 
    for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      for (int u = 0; u < CF; u++) {
	int uf = u * (PIXELS_PER_IMG/CF); 
	DATA_ITEM_TYPE v0 = 0.0f;
	DATA_ITEM_TYPE v1 = 0.0f;
      
 	DATA_ITEM_TYPE c0 = (r[tidx + uf + j] / g[tidx + uf + j] + (F0 + F1 * F1) * b[tidx + uf + j]) / (F1 * b[tidx + uf + j]); 
	DATA_ITEM_TYPE c1 =  F1 * b[tidx + uf + j];
//#pragma unroll UNROLL
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + c0;
	v1 = v0 - c1;
      }
      d_r[tidx + uf + j] = (r[tidx + uf + j] * v0 - g[tidx + uf + j] * F1 * v1);
      d_g[tidx + uf + j] = (g[tidx + uf + j] * F1 * (1.0f - v1) - r[tidx + uf + j]); 
      
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      d_b[tidx + uf + j] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_x[tidx + uf + j] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_a[tidx + uf + j] = v0;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_c[tidx + uf + j] = v1;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_d[tidx + uf + j] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_e[tidx + uf + j] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_f[tidx + uf + j] = v0;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_h[tidx + uf + j] = v1;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_j[tidx + uf + j] = v0;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_k[tidx + uf + j] = v1;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_l[tidx + uf + j] = v0;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_m[tidx + uf + j] = v1;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
    d_n[tidx + uf + j] = v1;
#endif
#if defined MEM16 || MEM17 || MEM18
    d_o[tidx + uf + j] = v1;
#endif
#if defined MEM17 || MEM18
    d_p[tidx + uf + j] = v1;
#endif
#if defined MEM18 
    d_q[tidx + uf + j] = v1;
#endif
      }
    }
  return;
}
#endif

#ifdef CA
__global__ void grayscale(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images) {
 
  
  //  size_t i = get_global_id(0);
  size_t local_id = threadIdx.x;
  size_t group_id = blockIdx.x;

  size_t tile_factor = (local_id / TILE) * TILE * (FIELDS - 1);  
  size_t thrd_offset;

  size_t loc_in_tile = (local_id * SPARSITY) % TILE;
  size_t offset_to_next_tile = (local_id / (TILE/SPARSITY)) * (FIELDS * TILE);

#if 1 //CA_OPT
  if (tile_factor == 0)
    thrd_offset = local_id + (group_id * WORKGROUP * FIELDS);
  else
    thrd_offset = local_id + tile_factor   + (group_id * WORKGROUP * FIELDS);
#else 
  int sets = (group_id / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  thrd_offset = loc_in_tile + offset_to_next_tile + (group_id - SPARSITY * sets) + (SPARSITY * FIELDS * TILE * sets);
#endif  


  //  size_t tile_factor = (threadIdx.x / TILE) * TILE * (FIELDS - 1);  
  //  size_t thrd_offset;
  
  //  thrd_offset = threadIdx.x + tile_factor + (group_id * WORKGROUP * FIELDS);

  float F0 = 0.02f; 
  float F1 = 0.30f; 

  size_t OR = OFFSET_R + thrd_offset;
  size_t OG = OFFSET_G + thrd_offset;
  size_t OB = OFFSET_B + thrd_offset;
  size_t OX = OFFSET_X + thrd_offset;
  size_t OA = OFFSET_A + thrd_offset;
  size_t OC = OFFSET_C + thrd_offset;
  size_t OD = OFFSET_D + thrd_offset;
  size_t OE = OFFSET_E + thrd_offset;
  size_t OF = OFFSET_F + thrd_offset;
  size_t OH = OFFSET_H + thrd_offset;
  size_t OJ = OFFSET_J + thrd_offset;
  size_t OK = OFFSET_K + thrd_offset;
  size_t OL = OFFSET_L + thrd_offset;
  size_t OM = OFFSET_M + thrd_offset;
  size_t ON = OFFSET_N + thrd_offset;
  size_t OO = OFFSET_O + thrd_offset;
  size_t OP = OFFSET_P + thrd_offset;
  size_t OQ = OFFSET_Q + thrd_offset;

  for (int p = 0; p < SWEEPS; p++) {
  for (int j = 0; j < NUM_IMGS * (FIELDS * PIXELS_PER_IMG); j = j + (PIXELS_PER_IMG * FIELDS)) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      
      DATA_ITEM_TYPE c0	= (src_images[OR + j] / src_images[OG + j] + (F0 + F1 * F1)
			   * src_images[OB + j]) / (F1 * src_images[OB + j]);
      DATA_ITEM_TYPE c1 = F1 * src_images[OB + j];
//#pragma unroll UNROLL
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + c0;
	v1 = v0 - c1;
      }
      dst_images[OR + j] = (src_images[OR + j] * v0
      					- src_images[OG + j] * F1 * v1);
      dst_images[OG + j] = (src_images[OG + j] * F1 * (1.0f - v1)
      					- src_images[OR + j]);
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OB + j] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OX + j] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OA + j] = v0; 
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OC + j]  = v1; 
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OD + j] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OE + j] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OF + j] = v0; 
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OH + j]  = v1; 
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OJ + j] = v0; 
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OK + j]  = v1; 
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OL + j] = v0; 
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OM + j]  = v1; 
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dst_images[ON + j]  = v1; 
#endif
#if defined MEM16 || MEM17 || MEM18
      dst_images[OO + j]  = v1; 
#endif
#if defined MEM17 || MEM18
      dst_images[OP + j]  = v1; 
#endif
#if defined MEM18
      dst_images[OQ + j]  = v1; 
#endif
  }
  }
  return;
}
#endif

#ifdef SOA
__global__ void grayscale(img *src_images, img *dst_images) {
   int tidx = threadIdx.x + blockDim.x * blockIdx.x;
   float F0 = 0.02f; 
   float F1 = 0.30f; 
   
  for (int j = 0; j < NUM_IMGS; j++) {
    DATA_ITEM_TYPE v0 = 0.0f;
    DATA_ITEM_TYPE v1 = 0.0f;
      DATA_ITEM_TYPE c0 = (src_images[tidx + j].r / src_images[tidx + j].g + (F0 + F1 * F1) * 
      	      src_images[tidx + j].b) / (F1 * src_images[tidx + j].b);
      DATA_ITEM_TYPE c1 = F1 * src_images[tidx + j].b;
#pragma unroll UNROLL
    for (int k = 0; k < ITERS; k++) {
      v0 = v0 + c0;
      v1 = v0 - c1;
    }
    dst_images[j].r[tidx] = (src_images[j].r[tidx] * v0 - src_images[j].g[tidx] * F1 * v1);
    dst_images[j].g[tidx]  = (src_images[j].g[tidx] * F1 * (1.0f - v1) - src_images[j].r[tidx]); 
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].b[tidx] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].x[tidx] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].a[tidx] = v0; 
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].c[tidx] = v1; 
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].d[tidx] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].e[tidx] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].f[tidx] = v0; 
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].h[tidx] = v1; 
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].j[tidx] = v0; 
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].k[tidx] = v1; 
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].l[tidx] = v0; 
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].m[tidx] = v1; 
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].n[tidx] = v1; 
#endif
#if defined MEM16 || MEM17 || MEM18
      dst_images[j].o[tidx] = v1; 
#endif
#if defined MEM17 || MEM18
      dst_images[j].p[tidx] = v1; 
#endif
#if defined MEM18
      dst_images[j].q[tidx] = v1; 
#endif
  }

}
#endif

void *host_grayscale_aos(void *p) {

#if 0
  pixel *src_images = ((args_aos *) p)->src;
  pixel *dst_images = ((args_aos *) p)->dst;
  int start =  ((args_aos *) p)->start_index;
  int end =  ((args_aos *) p)->end_index;
  DATA_ITEM_TYPE gs;
  for (int k = 0; k < ITERS; k++) {
    for (int j = start * PIXELS_PER_IMG; j < end * PIXELS_PER_IMG; j += PIXELS_PER_IMG) 
      for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) { 
	gs = (0.3 * src_images[i].r + 0.59 *
	      src_images[i].g + 0.11 * src_images[i].b + 1.0 *
	      src_images[i].x);
	dst_images[i].r = gs;
	dst_images[i].g = gs;
	dst_images[i].b = gs;
	dst_images[i].x = gs;
      }
  }
#endif
  return NULL;
}

void *host_grayscale_da(void *p) {

  DATA_ITEM_TYPE *r = ((args_da *) p)->r;
  DATA_ITEM_TYPE *g = ((args_da *) p)->g;
  DATA_ITEM_TYPE *b = ((args_da *) p)->b;
  DATA_ITEM_TYPE *x = ((args_da *) p)->x;
  DATA_ITEM_TYPE *d_r = ((args_da *) p)->d_r;
  DATA_ITEM_TYPE *d_g = ((args_da *) p)->d_g;
  DATA_ITEM_TYPE *d_b = ((args_da *) p)->d_b;
  DATA_ITEM_TYPE *d_x = ((args_da *) p)->d_x;
  int start =  ((args_da *) p)->start_index;
  int end =  ((args_da *) p)->end_index;

  DATA_ITEM_TYPE gs;
  for (int k = 0; k < ITERS; k++) {
    for (int j = start * PIXELS_PER_IMG; j < end * PIXELS_PER_IMG; j += PIXELS_PER_IMG) 
      for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) { 
	gs = (0.3 * r[i] + 0.59 * g[i] + 0.11 * b[i] + 1.0 * x[i]);
	d_r[i] = gs;
	d_g[i] = gs;
	d_b[i] = gs;
	d_x[i] = gs;
      }
  }
  return NULL;
}
void check_results_aos(pixel *src_images, pixel *dst_images, int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
#ifdef HOST
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
	      src_images[i].b) / (F1 * src_images[i].b);
	v1 = v0 - F1 * src_images[i].b;
      }
      DATA_ITEM_TYPE exp_result = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 

#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[512].r);
#endif
      float delta = fabs(dst_images[i].r - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef VERBOSE
        printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#else
  for (int j = 0; j < device_end * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
		   src_images[i].b) / (F1 * src_images[i].b);
	v1 = v0 - F1 * src_images[i].b;
      }
      DATA_ITEM_TYPE exp_result = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 

#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[i].r);
#endif
      float delta = fabs(dst_images[i].r - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
#endif
}

void check_results_da(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,
                      DATA_ITEM_TYPE *d_r, int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
#ifdef HOST
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG) {
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
	v1 = v0 - F1 * b[i];
      }
      DATA_ITEM_TYPE exp_result = (r[i] * v0 - g[i] * F1 * v1);
#ifdef DEBUG
	if (i == 512)
	  printf("%3.2f %3.2f\n", exp_result, d_r[i]);
#endif
      float delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
	  errors++;
#ifdef VERBOSE
	  if (errors < MAX_ERRORS) 
	    printf("%d %f %f\n", i, exp_result, d_r[i]);
#endif
	}
    }
  }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#endif

#ifdef DEVICE 
  for (int j = 0; j < device_end * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
	v1 = v0 - F1 * b[i];
      }
      DATA_ITEM_TYPE exp_result = (r[i] * v0 - g[i] * F1 * v1);

#ifdef DEBUG
      if (i == 512)    // check a pixel in the middle of the image
        printf("%f %f\n", exp_result, d_r[i]);
#endif
      float delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%f %f\n", exp_result, d_r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
  #endif 
  return;
}

void check_results_ca(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images, 
                      int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
      for (int k = j; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS) {
	for (int m = k; m < k + TILE; m++) { 
	  DATA_ITEM_TYPE v0 = 0.0f;
	  DATA_ITEM_TYPE v1 = 0.0f;
	  for (int k = 0; k < ITERS; k++) {
	    v0 = v0 + (src_images[OFFSET_R + m] / src_images[OFFSET_G + m] + (F0 + F1 * F1) 
		       * src_images[OFFSET_B + m]) / (F1 * src_images[OFFSET_B + m]);
	    v1 = v0 - F1 * src_images[OFFSET_B + m];
	  }
	  DATA_ITEM_TYPE exp_result = 
	    (src_images[OFFSET_R + m] * v0 - src_images[OFFSET_G + m] * F1 * v1);
#ifdef DEBUG
	  if (m == 0)
	    printf("%f %f\n", exp_result, dst_images[OFFSET_R + m]);
#endif
	  float delta = fabs(dst_images[OFFSET_R + m] - exp_result);
	  if (delta/exp_result > ERROR_THRESH) {
	    errors++;
#ifdef DEBUG
	    if (errors < MAX_ERRORS)
	      printf("%d %f %f\n", m, exp_result, dst_images[m]);
#endif
	  }
	}
      }

#if 0
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[OFFSET_R + i] / src_images[OFFSET_G + i] + (F0 + F1 * F1) 
		   * src_images[OFFSET_B + i]) / (F1 * src_images[OFFSET_B + i]);
	v1 = v0 - F1 * src_images[OFFSET_B + i];
      }
      DATA_ITEM_TYPE exp_result = (src_images[OFFSET_R + i] * v0 - src_images[OFFSET_G + i] * F1 * v1);
#ifdef DEBUG
      if (i == 512)
      printf("%f %f\n", exp_result, dst_images[OFFSET_B + i]);
      printf("%f %f %f\n", dst_images[OFFSET_R + i], dst_images[OFFSET_G + i], dst_images[OFFSET_B + i]);
#endif
      float delta = fabs(dst_images[OFFSET_R + i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%f %f\n", exp_result, dst_images[i]);
#endif
      }
    }
#endif
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


void check_results_soa(img *src_images, img *dst_images, int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
  for (int j = 0; j < device_end; j++) 
    for (unsigned int i = 0; i < PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[j].r[i] / src_images[j].g[i] + (F0 + F1 * F1) * 
		   src_images[j].g[i]) / (F1 * src_images[j].g[i]);
	v1 = v0 - F1 * src_images[j].g[i];
      }
      DATA_ITEM_TYPE exp_result = (src_images[j].r[i] * v0 - src_images[j].g[i] * F1 * v1);
#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[j].r[i]);
#endif
      float delta = fabs(dst_images[j].r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
          printf("%d %f %f\n", i, exp_result, dst_images[j].r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


int main(int argc,char *argv[]) {
  

#ifdef AOS
#ifdef UM
  pixel *src_images;
  gpuErrchk(cudaMallocManaged(&src_images, sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG))
#else
  pixel *src_images = (pixel *) malloc((sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG)); 
#endif

  pixel *dst_images = (pixel *) malloc(sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG); 


  /* 
   * Initialization 
   */
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
   for (int i = j; i < j + PIXELS_PER_IMG; i++) {
     src_images[i].r = (DATA_ITEM_TYPE)i;
     src_images[i].g = i * 10.0f;
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].b = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].x = i * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].a = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].c = i * 10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].d = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].e = i * 10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].f = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].h = i * 10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].j = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].k = i * 10.0f;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].l = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].m = i * 10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].n = i * 10.0f;
#endif
#if defined MEM16 || MEM17 || MEM18
	  src_images[i].o = i * 10.0f;
#endif
#if defined MEM17 || MEM18
	  src_images[i].p = i * 10.0f;
#endif
#if defined MEM18
	  src_images[i].q = i * 10.0f;
#endif
   }


  /* Copy for verification of results on CPU;  needed for C2GI pattern but doing it for other patterns as well  */
  pixel *src_images_copy = (pixel *) malloc((sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG)); 
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
   for (int i = j; i < j + PIXELS_PER_IMG; i++) {
     src_images_copy[i].r = src_images[i].r;
     src_images_copy[i].g = src_images[i].g; 
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
     src_images_copy[i].b = src_images[i].b;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
     src_images_copy[i].x = src_images[i].x;
#endif
   }


  pixel *d_src_images;
  pixel *d_dst_images;
#endif

#ifdef DA
  DATA_ITEM_TYPE *r = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *g = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *b = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *x = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *a = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *c = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *d = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *e = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *f = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *h = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *j_data = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *k_data = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *l = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *m = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *n = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *o = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *p = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *q = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 

  DATA_ITEM_TYPE *dst_r = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_g = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_b = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_x = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_a = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_c = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_d = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_e = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_f = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_h = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_j_data = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_k_data = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_l = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_m = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_n = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_o = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_p = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_q = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
    for (int i = j; i < j + PIXELS_PER_IMG; i++) {
      r[i] = (DATA_ITEM_TYPE)i;
      g[i] = i * 10.0f;
      b[i] = (DATA_ITEM_TYPE)i;
      x[i] = i * 10.0f;
      a[i] = (DATA_ITEM_TYPE)i;
      c[i] = i * 10.0f;
      d[i] = (DATA_ITEM_TYPE)i;
      e[i] = i * 10.0f;
      f[i] = (DATA_ITEM_TYPE)i;
      h[i] = i * 10.0f;
      j_data[i] = (DATA_ITEM_TYPE)i;
      k_data[i] = i * 10.0f;
      l[i] = (DATA_ITEM_TYPE)i;
      m[i] = i * 10.0f;
      n[i] = (DATA_ITEM_TYPE)i;
      o[i] = i * 10.0f;
      p[i] = (DATA_ITEM_TYPE)i;
      q[i] = i * 10.0f;
  }

  DATA_ITEM_TYPE *d_r, *d_g, *d_b, *d_x, *d_a, *d_c, *d_d, *d_e, *d_f, *d_h, *d_j, *d_k, *d_l, *d_m, *d_n, *d_o, *d_p, *d_q;
  DATA_ITEM_TYPE *d_dst_r, *d_dst_g, *d_dst_b, *d_dst_x, *d_dst_a, *d_dst_c, *d_dst_d, *d_dst_e, *d_dst_f, *d_dst_h, *d_dst_j, *d_dst_k, *d_dst_l, *d_dst_m, *d_dst_n, *d_dst_o, *d_dst_p, *d_dst_q;
#endif

#ifdef CA

  unsigned long size = NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
  DATA_ITEM_TYPE *src_images;
  DATA_ITEM_TYPE *dst_images;

  src_images = (DATA_ITEM_TYPE *) malloc(size);
  dst_images = (DATA_ITEM_TYPE *) malloc(size);

  if (!src_images) {
    printf("Unable to malloc src_images to fine grain memory. Exiting\n");
    exit(0);
  }
  if (!dst_images) {
    printf("Unable to malloc dst_images to fine grain memory. Exiting\n");
    exit(0);
  }

  int loc_sparsity = SPARSITY;
  int this_set_tile_count = 0;
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
    for (int k = j, t = 0; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS, t++) {
	for (int m = k, n = 0; m < k + TILE; m++, n++) { 	
	  if (t == loc_sparsity) {
	    this_set_tile_count++;
	    t = 0;
	  }
	  int val = (n * loc_sparsity + t) + (this_set_tile_count * loc_sparsity * TILE);
	  //    for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	  src_images[OFFSET_R + m] = (DATA_ITEM_TYPE)val;
	  src_images[OFFSET_G + m] = val *10.0f;
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_B + m] = (DATA_ITEM_TYPE)val;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_X + m] = val * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_A + m] = (DATA_ITEM_TYPE)val;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_C + m] = val * 10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_D + m] = (DATA_ITEM_TYPE)val;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_E + m] = val * 10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_F + m] = (DATA_ITEM_TYPE) val;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_H + m] = val * 10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_J + m] = (DATA_ITEM_TYPE) val;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_M + m] = val * 10.0f;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_L + m] = (DATA_ITEM_TYPE) val;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_M + m] = val * 10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      src_images[OFFSET_N + m] = (DATA_ITEM_TYPE)val;
#endif
#if defined MEM16 || MEM17 || MEM18
      src_images[OFFSET_O + m] = val * 10.0f;
#endif
#if defined MEM17 || MEM18
      src_images[OFFSET_P + m] = (DATA_ITEM_TYPE)val;
#endif
#if defined MEM18
      src_images[OFFSET_Q + m] = val * 10.0f;
#endif
	}
    }


  DATA_ITEM_TYPE *d_src_images;
  DATA_ITEM_TYPE *d_dst_images;
#endif

#ifdef SOA
  img *src_images;
  img *dst_images;

  src_images = (img *) malloc(sizeof(img) * NUM_IMGS);
  dst_images = (img *) malloc(sizeof(img) * NUM_IMGS);

  if (!src_images) {
    printf("Unable to malloc src_images to fine grain memory. Exiting\n");
    exit(0);
  }
  if (!dst_images) {
    printf("Unable to malloc dst_images to fine grain memory. Exiting\n");
    exit(0);
  }
  for (int j = 0; j < NUM_IMGS; j++) 
    for (int k = 0; k < PIXELS_PER_IMG; k++) {
      src_images[j].r[k] = (DATA_ITEM_TYPE)k;
      src_images[j].g[k] = k * 10.0f;
      src_images[j].b[k] = (DATA_ITEM_TYPE)k;
      src_images[j].x[k] = k * 10.0f;
    }

  img *d_src_images;
  img *d_dst_images;
#endif 

  int host_start = 0;
  int device_end = NUM_IMGS;

#ifdef HETERO 
  unsigned int imgs_per_device = NUM_IMGS/DEVICES;
  device_end = host_start = imgs_per_device;   // device [0..(N/2)], host [(N/2 + 1) ... N]
#endif 

  // layout change: copy src data into DA 
#ifdef COPY
  copy_aos_to_da(r, g, b, x, src_images);
#endif

  double t;
#ifdef HOST
  pthread_t threads[CPU_THREADS];
  unsigned int num_imgs_per_thread = ((NUM_IMGS - host_start)/CPU_THREADS);
  int start = host_start;

  t = mysecond();
#ifdef AOS
  args_aos host_args[CPU_THREADS];
#else
  args_da host_args[CPU_THREADS];
#endif
  for (int i = 0; i < CPU_THREADS; i++) {
    host_args[i].start_index = start;
    host_args[i].end_index = start + num_imgs_per_thread;
#ifdef AOS
    // entire array passed to each thread; should consider passing sections
    host_args[i].src = src_images;
    host_args[i].dst = dst_images;
    pthread_create(&threads[i], NULL, &host_grayscale_aos, (void *) &host_args[i]);
#else 
    host_args[i].r = r;
    host_args[i].g = g;
    host_args[i].b = b;
    host_args[i].x = x;
    host_args[i].d_r = dst_r;
    host_args[i].d_g = dst_g;
    host_args[i].d_b = dst_b;
    host_args[i].d_x = dst_x;
    pthread_create(&threads[i], NULL, &host_grayscale_da, (void *) &host_args[i]);
#endif 
    start = start + num_imgs_per_thread;
  }
#endif

#ifdef DEVICE 
  cudaEvent_t start_copy_to_dev , stop_copy_to_dev;
  cudaEvent_t start_copy_to_host , stop_copy_to_host;
  float msec_with_copy_to_dev = 0.0f;
  float msec_with_copy_to_host = 0.0f;

  gpuErrchk(cudaEventCreate(&start_copy_to_dev));
  gpuErrchk(cudaEventCreate(&stop_copy_to_dev));

  gpuErrchk(cudaEventCreate(&start_copy_to_host));
  gpuErrchk(cudaEventCreate(&stop_copy_to_host));


#ifdef AOS
#ifndef UM
  gpuErrchk(cudaMalloc((void **) &d_src_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel)));
#endif
  gpuErrchk(cudaMalloc((void **) &d_dst_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel)));

#ifndef UM
   gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
   gpuErrchk(cudaMemcpy(d_src_images,src_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel),cudaMemcpyHostToDevice));
   gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
#endif
#endif
#ifdef DA
  gpuErrchk(cudaMalloc((void **) &d_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_f, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_h, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_j, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_k, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_l, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_m, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_n, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_o, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_p, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_q, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));

  gpuErrchk(cudaMalloc((void **) &d_dst_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_f, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_h, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_j, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_k, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_l, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_m, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_n, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_o, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_p, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_q, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));

  gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
  gpuErrchk(cudaMemcpy(d_r, r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_g, g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_x, x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_a, a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_c, c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_d, d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_e, e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_f, f, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_h, h, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_j, j_data, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_k, k_data, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_l, l, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_m, m, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_n, n, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_o, o, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_p, p, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_q, q, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
  gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
#endif
#ifdef CA
  gpuErrchk(cudaMalloc((void **) &d_src_images, size));
  gpuErrchk(cudaMalloc((void **) &d_dst_images, size)); 
  gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
  gpuErrchk(cudaMemcpy(d_src_images,src_images, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
#endif 
#ifdef SOA
  gpuErrchk(cudaMalloc((void **) &d_src_images, sizeof(img) * NUM_IMGS));
  gpuErrchk(cudaMalloc((void **) &d_dst_images, sizeof(img) * NUM_IMGS)); 

  gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
  gpuErrchk(cudaMemcpy(d_dst_images,dst_images, sizeof(img) * NUM_IMGS, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_src_images,src_images, sizeof(img) * NUM_IMGS, cudaMemcpyHostToDevice))
  gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
#endif
  cudaEvent_t start_kernel , stop_kernel;
  float msecTotal = 0.0f;
  gpuErrchk(cudaEventCreate(&start_kernel));
  gpuErrchk(cudaEventCreate(&stop_kernel));
  
  gpuErrchk(cudaEventRecord(start_kernel,0));

 int threadsPerBlock = WORKGROUP;
 int blockPerGrid = THREADS/WORKGROUP; 

#ifdef C2GI
#ifdef AOS
 for (int i = 0; i < KERNEL_ITERS; i++) {
#ifdef UM
   grayscale<<<blockPerGrid,threadsPerBlock>>>(src_images, d_dst_images);
   cudaDeviceSynchronize();
   //   src_images = d_dst_images;
#else
   grayscale<<<blockPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);
   //   d_src_images = d_dst_images;
#endif
 }
#endif
#endif
 
#ifdef C2G
#ifdef AOS

#ifdef UM
   grayscale<<<blockPerGrid,threadsPerBlock>>>(src_images, d_dst_images);
#else
   grayscale<<<blockPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);
#endif

#endif
#endif
 
#ifdef DA
  grayscale<<<blockPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x, 
					      d_a, d_c, d_d, d_e, 
					      d_f, d_h, d_j, d_k, 
					      d_l, d_m, 
					      d_n, d_o, d_p, d_q, 
					      d_dst_r, d_dst_g, d_dst_b, d_dst_x, 
					      d_dst_a, d_dst_c, d_dst_d, d_dst_e, 
					      d_dst_f, d_dst_h, d_dst_j, d_dst_k, 
					      d_dst_l, d_dst_m,
					      d_dst_n, d_dst_o,
					      d_dst_p, d_dst_q);
#endif
#ifdef CA
  grayscale<<<blockPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);  
#endif
#ifdef SOA
  grayscale<<<blockPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);  
#endif
  gpuErrchk(cudaEventRecord(stop_kernel,0));

  gpuErrchk(cudaEventSynchronize(start_kernel));
  gpuErrchk(cudaEventSynchronize(stop_kernel));
  gpuErrchk(cudaEventElapsedTime(&msecTotal, start_kernel, stop_kernel));

#ifndef UM
   gpuErrchk(cudaEventSynchronize(start_copy_to_host));
   gpuErrchk(cudaEventSynchronize(stop_copy_to_host));
   gpuErrchk(cudaEventElapsedTime(&msec_with_copy_to_dev, start_copy_to_dev, stop_copy_to_dev));
#endif

#ifdef AOS
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel),cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef DA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_r, d_dst_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_g, d_dst_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_b, d_dst_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_x, d_dst_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_a, d_dst_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_c, d_dst_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_d, d_dst_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_e, d_dst_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_f, d_dst_f, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_h, d_dst_h, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_j_data, d_dst_j, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_k_data, d_dst_k, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_l, d_dst_l, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_m, d_dst_m, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_n, d_dst_n, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_m, d_dst_o, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_p, d_dst_p, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy(dst_q, d_dst_q, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef CA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, size, cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef SOA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, sizeof(img) * NUM_IMGS, cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif


  //  cudaDeviceSynchronize();
  //  gpuErrchk(cudaEventSynchronize(start_copy_to_host));
  //  gpuErrchk(cudaEventSynchronize(stop_copy_to_host));
  //  gpuErrchk(cudaEventElapsedTime(&msec_with_copy_to_host, start_copy_to_host, stop_copy_to_host));
#endif
#ifdef HOST
  for (int i = 0; i < CPU_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  t = 1.0E6 * (mysecond() - t);
#endif


#ifdef COPY 
  check_results_aos(src_images, dst_images, host_start, device_end);
  check_results_da(r, g, b, x, d_r, host_start, device_end);
#else
#ifdef AOS
  check_results_aos(src_images_copy, dst_images, host_start, device_end);
#endif
#ifdef DA
  check_results_da(r, g, b, x, dst_r, host_start, device_end);
#endif
#ifdef CA
  check_results_ca(src_images, dst_images, host_start, device_end);
#endif
#ifdef SOA
   check_results_soa(src_images, dst_images, host_start, device_end);
#endif
#endif


  /* derive perf metrics */
  unsigned long dataMB = (NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * STREAMS)/(1024 * 1024);
  double flop = (6.0  * (float) ITERS * (float) NUM_IMGS * (float) PIXELS_PER_IMG);

#ifndef HETERO
#ifdef DEVICE 
  t = msecTotal * 1000;
  double t_with_copy = (msec_with_copy_to_dev + msec_with_copy_to_host) * 1000;
  double secs_with_copy = t_with_copy/1000000;
#endif
#endif

  double secs = t/1000000;

#ifdef VERBOSE
  fprintf(stdout, "Kernel execution time %3.2f ms\n", t/1000);
  fprintf(stdout, "Attained bandwidth: %3.2f MB/s\n", dataMB/secs); 
  fprintf(stdout, "MFLOPS: %3.2f\n", (flop/secs)/(1024 * 1024)); 
#ifndef HETERO
#ifdef DEVICE
  fprintf(stdout, "Kernel execution time with copy %3.2f ms\n", t_with_copy/1000);
  fprintf(stdout, "Attained bandwidth with copy: %3.2f MB/s\n", dataMB/secs_with_copy); 
  fprintf(stdout, "MFLOPS with copy: %3.2f\n", (flop/secs_with_copy)/(1024 * 1024)); 
#endif
#endif
#else 
#ifdef HOST
  fprintf(stdout, "%3.2f ", t/1000);
  fprintf(stdout, "%3.2f ", dataMB/secs); 
  fprintf(stdout, "%3.2f\n ", (flop/secs)/(1024 * 1024)); 
#endif
#ifndef HETERO
#ifdef DEVICE
  fprintf(stdout, "%3.2f,", t/1000);
  fprintf(stdout, "%3.2f\n", (t_with_copy/1000));
//fprintf(stdout, "%3.2f\n", (t_with_copy/1000 - t/1000));
#endif
#endif
#endif

  // move device reset at the very end. if host touches cudaMallocManaged() codes 
  gpuErrchk(cudaDeviceReset());

  return 0;
}

