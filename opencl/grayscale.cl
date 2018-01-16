#include "dlbench.h"

#define EXP(a) exp(a)
#define LOG(a) log(a)
#define SQRT(a) sqrt(a)

///////////////////////////////////////////////////////////////////////////////
// Predefine functions to avoid bug in OpenCL compiler on Mac OSX 10.7 systems
///////////////////////////////////////////////////////////////////////////////
float CND(float d);
void BlackScholesBody(__global float *call, __global float *put,  float S,
					  float X, float T, float R, float V);

///////////////////////////////////////////////////////////////////////////////
// Rational approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
float CND(float d){
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
        K = 1.0f / (1.0f + 0.2316419f * fabs(d));

    float
        cnd = RSQRT2PI * EXP(- 0.5f * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


__kernel void grayscale_aos(__global pixel *src_images, __global pixel *dst_images, int num_imgs) {

  size_t i = get_global_id(0);

  float F0 = 0.02f; 
  float F1 = 0.30f; 

  for (int k = 0; k < ITERS; k++) {
    for (int j = 0; j < num_imgs * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      float v0 = (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
		  src_images[i].b) / (F1 * src_images[i].b);
      float v1 = v0 - F1 * src_images[i].b;

      dst_images[i].r = (src_images[i].r * v0 - src_images[i].g * F1 * v1);
      dst_images[i].g  = (src_images[i].g * F1 * (1.0f - v1) - src_images[i].r); 
      dst_images[i].b = v0;
      dst_images[i].x = v1;
    }
  }
}

__kernel void grayscale_da(__global DATA_ITEM_TYPE *r, __global DATA_ITEM_TYPE *g, 
			   __global DATA_ITEM_TYPE *b,
			    __global DATA_ITEM_TYPE *x,  
			   __global DATA_ITEM_TYPE *d_r, 
			   __global DATA_ITEM_TYPE *d_g, 
			   __global DATA_ITEM_TYPE *d_b, 
			   __global DATA_ITEM_TYPE *d_x,
			   int num_imgs) {

  size_t i = get_global_id(0);

  float F0 = 0.02f; 
  float F1 = 0.30f; 

  for (int k = 0; k < ITERS; k++) {
    for (int j = 0; j < num_imgs * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
      float v0 = (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
      float v1 = v0 - F1 * b[i];
      
      d_r[i] = (r[i] * v0 - g[i] * F1 * v1);
      d_g[i] = (g[i] * F1 * (1.0f - v1) - r[i]); 
      d_b[i] = v0;
      d_x[i] = v1;
   }
  }
}

__kernel void grayscale_aos_new(__global pixel *src_images, __global pixel *dst_images, int num_imgs) {

  float R = 0.02f; 
  float V = 0.30f; 
  size_t opt = get_global_id(0); 
  if (opt < num_imgs) {
    float sqrtT = SQRT(src_images[opt].b);
    float    d1 = (LOG(src_images[opt].r / src_images[opt].g) + (R + 0.5f * V * V) 
		   * src_images[opt].b) / (V * sqrtT);
    float    d2 = d1 - V * sqrtT;
    float CNDD1 = CND(d1);
    float CNDD2 = CND(d2);
    
    //Calculate Call and Put simultaneously
    float expRT = EXP(- R * src_images[opt].b);
    dst_images[opt].r = (src_images[opt].r * CNDD1 - src_images[opt].g * expRT * CNDD2);
    dst_images[opt].g  = (src_images[opt].g * expRT * (1.0f - CNDD2) - src_images[opt].r * (1.0f - CNDD1));
  }
}


__kernel void grayscale_soa(__global img *src_images,  __global img *dst_images, int num_imgs) {
  size_t i = get_global_id(0);
  DATA_ITEM_TYPE gs;
  for (int k = 0; k < ITERS; k++) {
    for (int j = 0; j < num_imgs; j++) {
      gs = (0.3 * src_images[j].r[i] + 0.59 * src_images[j].g[i] 
	    + 0.11 * src_images[j].b[i] + 1.0 * src_images[j].x[i]);
      dst_images[j].r[i] = gs;
      dst_images[j].g[i] = gs;
      dst_images[j].b[i] = gs;
      dst_images[j].x[i] = gs;
    }
  }
}


__kernel void grayscale_ca(__global DATA_ITEM_TYPE *src_images, __global DATA_ITEM_TYPE *dst_images, 
			   int num_imgs) {
  size_t i = get_global_id(0);
  DATA_ITEM_TYPE gs;
  for (int k = 0; k < ITERS; k++) {
    for (int j = 0; j < num_imgs * (4 * PIXELS_PER_IMG); j = j + (PIXELS_PER_IMG * 4)) {
      gs = (0.3 * src_images[OFFSET_R + i + j] + 0.59 * src_images[OFFSET_G + i + j]
            + 0.11 * src_images[OFFSET_B + i + j] + 1.0 * src_images[OFFSET_X + i + j]);
      dst_images[OFFSET_R + i + j] = gs;
      dst_images[OFFSET_G + i + j] = gs;
      dst_images[OFFSET_B + i + j] = gs;
      dst_images[OFFSET_X + i + j] = gs;
    }
  }
}
