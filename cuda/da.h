#include<kernel.h>
#ifdef DA
#if (MEM == 1)
__global__ void grayscale(float *r, 
			  float *dst_r) {
#endif
#if (MEM == 2)
__global__ void grayscale(float *r, float *g, 
			  float *dst_r, float *dst_g) {
#endif
#if (MEM == 3)
__global__ void grayscale(float *r, float *g, float *b, 			  
			  float *dst_r, float *dst_g, float *dst_b) {
#endif
#if (MEM == 4)
__global__ void grayscale(float *r, float *g, float *b, float *x,  			  
			  float *dst_r, float *dst_g, float *dst_b, float *dst_x) {
#endif
#if (MEM == 5)
__global__ void grayscale(float *r, float *g, float *b, float *x, float *a, 			  
			  float *dst_r, float *dst_g, float *dst_b, float *dst_x, float *dst_a) {
#endif
#if (MEM == 6)
__global__ void grayscale(float *r, float *g, float *b, float *x, float *a, float *c, 			  
			  float *dst_r, float *dst_g, float *dst_b, float *dst_x, float *dst_a, float *dst_c) {
#endif
#if (MEM == 7)
__global__ void grayscale(float *r, float *g, float *b, float *x, float *a, float *c, float *d, 			  
			  float *dst_r, float *dst_g, float *dst_b, float *dst_x, float *dst_a, float *dst_c, float *dst_d) {
#endif
#if (MEM == 8)
  __global__ void grayscale(float *r, float *g, float *b, float *x, float *a, float *c, float *d, float *e, 
			    float *dst_r, float *dst_g, float *dst_b, float *dst_x, float *dst_a, float *dst_c, float *dst_d,
			    float *dst_e) {
#endif
  int tidx = threadIdx.x;// + blockDim.x * blockIdx.x;

#if 1
  //  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  int sets = (blockIdx.x / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  tidx = (tidx * SPARSITY) + (blockIdx.x - SPARSITY * sets) + set_offset;
#endif

  DATA_ITEM_TYPE alpha = 0.0f;
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
