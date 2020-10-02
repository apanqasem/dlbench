#include<kernel.h>
#ifdef DA
#if (MEM == 1)
__global__ void grayscale(DATA_ITEM_TYPE *r, 
			  DATA_ITEM_TYPE *dst_r) {
#endif
#if (MEM == 2)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, 
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g) {
#endif
#if (MEM == 3)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, 			  
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b) {
#endif
#if (MEM == 4)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,  			  
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b, DATA_ITEM_TYPE *dst_x) {
#endif
#if (MEM == 5)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x, DATA_ITEM_TYPE *a, 			  
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b, DATA_ITEM_TYPE *dst_x, DATA_ITEM_TYPE *dst_a) {
#endif
#if (MEM == 6)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x, DATA_ITEM_TYPE *a, DATA_ITEM_TYPE *c, 			  
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b, DATA_ITEM_TYPE *dst_x, DATA_ITEM_TYPE *dst_a, DATA_ITEM_TYPE *dst_c) {
#endif
#if (MEM == 7)
__global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x, DATA_ITEM_TYPE *a, DATA_ITEM_TYPE *c, DATA_ITEM_TYPE *d, 			  
			  DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b, DATA_ITEM_TYPE *dst_x, DATA_ITEM_TYPE *dst_a, DATA_ITEM_TYPE *dst_c, DATA_ITEM_TYPE *dst_d) {
#endif
#if (MEM == 8)
  __global__ void grayscale(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x, DATA_ITEM_TYPE *a, DATA_ITEM_TYPE *c, DATA_ITEM_TYPE *d, DATA_ITEM_TYPE *e, 
			    DATA_ITEM_TYPE *dst_r, DATA_ITEM_TYPE *dst_g, DATA_ITEM_TYPE *dst_b, DATA_ITEM_TYPE *dst_x, DATA_ITEM_TYPE *dst_a, DATA_ITEM_TYPE *dst_c, DATA_ITEM_TYPE *dst_d,
			    DATA_ITEM_TYPE *dst_e) {
#endif
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;

#if 0
  int sets = (blockIdx.x / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  int tidx = (tidx * SPARSITY) + (blockIdx.x - SPARSITY * sets) + set_offset;
#endif

  DATA_ITEM_TYPE alpha = 0.0;
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {

#if (MEM == 1)  
    KERNEL2(alpha,r[tidx + j],r[tidx + j],r[tidx + j]);
#endif
#if (MEM == 2) 
    KERNEL2(alpha,r[tidx + j],g[tidx + j],g[tidx + j]);
#endif
#if (MEM > 2) 
    KERNEL2(alpha,r[tidx + j],g[tidx + j],b[tidx + j]);
#endif 
    for (int k = 0; k < ITERS; k++)
      KERNEL1(alpha,alpha,r[tidx + j]);

#if (MEM >= 1)     
    dst_r[tidx + j] = alpha;
#endif
#if (MEM >= 2)     
    dst_g[tidx + j] = alpha;
#endif
#if (MEM >= 3)     
    dst_b[tidx + j] = alpha;
#endif
#if (MEM >= 4)
    dst_x[tidx + j] = x[tidx + j];
#endif
#if (MEM >= 5)
    dst_a[tidx + j] = a[tidx + j];
#endif
#if (MEM >= 6)
    dst_c[tidx + j]  = c[tidx + j];
#endif
#if (MEM >= 7)
    dst_d[tidx + j] = d[tidx + j];
#endif
#if (MEM >= 8)
    dst_e[tidx + j] = e[tidx + j];
#endif
  }
}
#endif
