#include "dlbench.h"

__kernel void grayscale_aos(__global pixel *src_images, 
			    __global pixel *dst_images, 
			    int num_imgs) {

  size_t i = get_local_id(0); 
  size_t group_id = get_group_id(0);

#if 0
  int sets = (group_id / SPARSITY);    // sets processed
  int set_offset = WORKGROUP * SPARSITY * sets;
  i = (i * SPARSITY) + (group_id - SPARSITY * sets) + set_offset;
#endif
  float F0 = 0.02f; 
  float F1 = 0.30f; 
  
#define BLK 20
    //       for (int jj = 0; jj < num_imgs * PIXELS_PER_IMG; jj = jj + (PIXELS_PER_IMG * BLK)) {
    for (int p = 0; p < SWEEPS; p++) {
      for (int j = 0; j < num_imgs * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
	//	for (int j = jj; j < jj + (PIXELS_PER_IMG * BLK); j = j + PIXELS_PER_IMG) {
	for (int k = 0; k < PIXELS_PER_IMG/THREADS; k++) {
	  DATA_ITEM_TYPE v0 = 0.0f;
	  DATA_ITEM_TYPE v1 = 0.0f;
	  
	  DATA_ITEM_TYPE c1 = (src_images[i + j + k].r / src_images[i + j + k].g + (F0 + F1 * F1) *
	  		       src_images[i + j + k].b) / (F1 * src_images[i + j + k].b);
	  DATA_ITEM_TYPE c2 = F1 * src_images[i + j + k].b;

	  for (int k = 0; k < ITERS; k++) {
	    v0 = v0 + c1;
	    v1 = v0 - c2;
	  }
	  dst_images[i + j + k].r = (src_images[i + j + k].r * v0 
				     - src_images[i + j + k].g * F1 * v1);
	  dst_images[i + j + k].g  = (src_images[i + j + k].g * F1 * (1.0f - v1) 
				      - src_images[i + j + k].r);
	  
	  dst_images[i + j + k].b = v0;
#if (MEM >= 4)
	  dst_images[i + j + k].x = v1;
#endif
#if (MEM >= 5)
	  dst_images[i + j + k].a = v0;
#endif
#if (MEM >= 6)
	  dst_images[i + j + k].c  = v1;
#endif
#if (MEM >= 7)
	  dst_images[i + j + k].d = v0;
#endif
#if (MEM >= 8)
	  dst_images[i + j + k].e = v1;
#endif
#if (MEM >= 9)
	  dst_images[i + j + k].f = v0;
#endif
#if (MEM >= 10)
	  dst_images[i + j + k].h  = v1;
#endif
#if (MEM >= 11)
	  dst_images[i + j + k].j = v0;
#endif
#if (MEM >= 12)
	  dst_images[i + j + k].k  = v1;
#endif
#if (MEM >= 13)
	  dst_images[i + j + k].l = v0;
#endif
#if (MEM >= 14)
	  dst_images[i + j + k].m  = v1;
#endif
#if (MEM >= 15)
	  dst_images[i + j + k].n  = v1;
#endif
#if (MEM >= 16)
	  dst_images[i + j + k].o  = v1;
#endif
#if (MEM >= 17)
	  dst_images[i + j + k].p  = v1;
#endif
#if (MEM >= 18)
	  dst_images[i + j + k].q  = v1;
#endif
	}
      }
    }
}
    
__kernel void copy_da(__global DATA_ITEM_TYPE *r, __global DATA_ITEM_TYPE *d_r, 
		      __global DATA_ITEM_TYPE *dev_r, __global DATA_ITEM_TYPE *dev_d_r, 

		      int num_imgs) {

  const size_t i = get_local_id(0);
  d_r[i] = r[i];
  
}
#if 0
__kernel void grayscale_da_new(__global DATA_ITEM_TYPE *r, __global DATA_ITEM_TYPE *g, 
			       __global DATA_ITEM_TYPE *b, 
			       __global DATA_ITEM_TYPE *d_r, 
			       __global DATA_ITEM_TYPE *d_g, 
			       __global DATA_ITEM_TYPE *d_b, 
			       int num_imgs) {

  size_t i = get_local_id(0);
  size_t group_id = get_group_id(0);

#if 0
  int sets = (group_id / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  i = (i * SPARSITY) + (group_id - SPARSITY * sets) + set_offset;
#endif

  float F0 = 0.02f; 
  float F1 = 0.30f; 

  for (int p = 0; p < SWEEPS; p++) {
    for (int j = 0; j < num_imgs * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      //      for (int u = 0; u < CF; u++) {
      for (int k = 0; k < PIXELS_PER_IMG/THREADS; k++) {

	//	  int uf = 0;// u * PIXELS_PER_IMG/CF; 
	/* DATA_ITEM_TYPE r_val = r[i + uf + j]; */
	/* DATA_ITEM_TYPE g_val = g[i + uf + j]; */
	/* DATA_ITEM_TYPE b_val = b[i + uf + j]; */
	DATA_ITEM_TYPE v0 = 0.0f;
	DATA_ITEM_TYPE v1 = 0.0f;

	DATA_ITEM_TYPE c1 = (r[i + k + j] / g[i + k + j] + (F0 + F1 * F1) *
			     b[i + k + j]) / (F1 * b[i + k + j]);
	DATA_ITEM_TYPE c2 = F1 * b[i + k + j];

	//#pragma unroll UNROLL
	for (int k = 0; k < ITERS; k++) {
	  v0 = v0 + c1;
	  v1 = v0 - c2;
	}

	d_r[i + k + j] = (r[i + k + j] * v0 - g[i + k + j] * F1 * v1);
	d_g[i + k + j] = (g[i + k + j] * F1 * (1.0f - v1) - r[i + k + j]);

#if (MEM == 2 || MEM == 3) 
	d_b[i + k + j] = v0;
#endif
      }
    }
  } 
}
#endif
#if 0
__kernel void grayscale_da(__global DATA_ITEM_TYPE *r, __global DATA_ITEM_TYPE *g, 
			   __global DATA_ITEM_TYPE *b, __global DATA_ITEM_TYPE *x,  
			   __global DATA_ITEM_TYPE *a, __global DATA_ITEM_TYPE *c, 
			   __global DATA_ITEM_TYPE *d, __global DATA_ITEM_TYPE *e,  
			   __global DATA_ITEM_TYPE *f, __global DATA_ITEM_TYPE *h,  
			   __global DATA_ITEM_TYPE *j, __global DATA_ITEM_TYPE *k,  
			   __global DATA_ITEM_TYPE *l, __global DATA_ITEM_TYPE *m,  
			   __global DATA_ITEM_TYPE *n, __global DATA_ITEM_TYPE *o,  
			   __global DATA_ITEM_TYPE *p, __global DATA_ITEM_TYPE *q,  
			   __global DATA_ITEM_TYPE *d_r, 
			   __global DATA_ITEM_TYPE *d_g, 
			   __global DATA_ITEM_TYPE *d_b, 
			   __global DATA_ITEM_TYPE *d_x,
			   __global DATA_ITEM_TYPE *d_a, 
			   __global DATA_ITEM_TYPE *d_c, 
			   __global DATA_ITEM_TYPE *d_d, 
			   __global DATA_ITEM_TYPE *d_e,
			   __global DATA_ITEM_TYPE *d_f, 
			   __global DATA_ITEM_TYPE *d_h,
			   __global DATA_ITEM_TYPE *d_j, 
			   __global DATA_ITEM_TYPE *d_k,
			   __global DATA_ITEM_TYPE *d_l, 
			   __global DATA_ITEM_TYPE *d_m,
			   __global DATA_ITEM_TYPE *d_n, __global DATA_ITEM_TYPE *d_o,  
			   __global DATA_ITEM_TYPE *d_p, __global DATA_ITEM_TYPE *d_q,  
			   int num_imgs) {

  size_t i = get_local_id(0);
  size_t group_id = get_group_id(0);

  int sets = (group_id / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  i = (i * SPARSITY) + (group_id - SPARSITY * sets) + set_offset;


  float F0 = 0.02f; 
  float F1 = 0.30f; 

  for (int p = 0; p < SWEEPS; p++) {
    for (int j = 0; j < num_imgs * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      //      for (int u = 0; u < CF; u++) {
      int uf = 0;// u * PIXELS_PER_IMG/CF; 
	/* DATA_ITEM_TYPE r_val = r[i + uf + j]; */
	/* DATA_ITEM_TYPE g_val = g[i + uf + j]; */
	/* DATA_ITEM_TYPE b_val = b[i + uf + j]; */
	DATA_ITEM_TYPE v0 = 0.0f;
	DATA_ITEM_TYPE v1 = 0.0f;

	DATA_ITEM_TYPE c1 = (r[i + uf + j] / g[i + uf + j] + (F0 + F1 * F1) *
			     b[i + uf + j]) / (F1 * b[i + uf + j]);
	DATA_ITEM_TYPE c2 = F1 * b[i + uf + j];

#pragma unroll UNROLL
	for (int k = 0; k < ITERS; k++) {
	  v0 = v0 + c1;
	  v1 = v0 - c2;
	}

	d_r[i + uf + j] = (r[i + uf + j] * v0 - g[i + uf + j] * F1 * v1);
	d_g[i + uf + j] = (g[i + uf + j] * F1 * (1.0f - v1) - r[i + uf + j]);

#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	d_b[i + uf + j] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_x[i + uf + j] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_a[i + uf + j] = v0;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_c[i + uf + j] = v1;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_d[i + uf + j] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_e[i + uf + j] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_f[i + uf + j] = v0;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_h[i + uf + j] = v1;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_j[i + uf + j] = v0;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_k[i + uf + j] = v1;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_l[i + uf + j] = v0;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    d_m[i + uf + j] = v1;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
    d_n[i + uf + j] = v0;
#endif
#if defined MEM16 || MEM17 || MEM18
    d_o[i + uf + j] = v1;
#endif
#if defined MEM17 || MEM18
    d_p[i + uf + j] = v0;
#endif
#if defined MEM18
    d_q[i + uf + j] = v1;
#endif
    // }
    }
  }
}
#endif

__kernel void grayscale_ca(__global DATA_ITEM_TYPE *src_images, __global DATA_ITEM_TYPE *dst_images, 
			   int num_imgs) {

  size_t i = get_global_id(0);
  size_t local_id = get_local_id(0);
  size_t group_id = get_group_id(0);

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
    
  float F0 = 0.02f; 
  float F1 = 0.30f; 
  for (int p = 0; p < SWEEPS; p++) {
    for (int j = 0; j < num_imgs * (FIELDS * PIXELS_PER_IMG); j = j + (PIXELS_PER_IMG * FIELDS)) {
      DATA_ITEM_TYPE r = src_images[OFFSET_R + thrd_offset + j]; 
      DATA_ITEM_TYPE g = src_images[OFFSET_G + thrd_offset + j]; 
      DATA_ITEM_TYPE b = src_images[OFFSET_B + thrd_offset + j];
      
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      
      #pragma unroll UNROLL
      for (int k = 0; k < ITERS; k++) {
      	v0 = v0 + (r / g + (F0 + F1 * F1) * b) / (F1 * b); 
      	v1 = v0 - F1 * b; 
      }
      dst_images[OFFSET_R + thrd_offset + j] = (r * v0 - g * F1 * v1);
      dst_images[OFFSET_G + thrd_offset + j] = (g * F1 * (1.0f - v1) -r);
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_B + thrd_offset + j] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_X + thrd_offset + j] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_A + thrd_offset + j] = v0;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_C + thrd_offset + j] = v1;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_D + thrd_offset + j] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_E + thrd_offset + j] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_F + thrd_offset + j] = v0;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_H + thrd_offset + j] = v1;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_J + thrd_offset + j] = v0;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_K + thrd_offset + j] = v1;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_L + thrd_offset + j] = v0;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_M + thrd_offset + j] = v1;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dst_images[OFFSET_N + thrd_offset + j] = v0;
#endif
#if defined MEM16 || MEM17 || MEM18
      dst_images[OFFSET_O + thrd_offset + j] = v1;
#endif
#if defined MEM17 || MEM18
      dst_images[OFFSET_P + thrd_offset + j] = v0;
#endif
#if defined MEM18
      dst_images[OFFSET_Q + thrd_offset + j] = v1;
#endif
  }
  }
}

__kernel void grayscale_soa(__global img *src_images,  __global img *dst_images, int num_imgs) {

  size_t i = get_global_id(0);

  float F0 = 0.02f; 
  float F1 = 0.30f; 

  for (int j = 0; j < num_imgs; j++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[j].r[i] / src_images[j].g[i] + (F0 + F1 * F1) * 
		   src_images[j].g[i]) / (F1 * src_images[j].g[i]);
	v1 = v0 - F1 * src_images[j].g[i];
      }
    dst_images[j].r[i] = (src_images[j].r[i] * v0 - src_images[j].g[i] * F1 * v1);
    dst_images[j].g[i]  = (src_images[j].g[i] * F1 * (1.0f - v1) - src_images[j].r[i]); 
#if 0
    dst_images[j].b[i] = v0;
    dst_images[j].x[i] = v1;
#endif
  }
}


