#include <stdlib.h>
#include <host_kernels.h>
#include <dlbench.h>

void *host_grayscale_aos_pixel(void *p) {

  pixel *src_images = ((args_aos *) p)->src;
  pixel *dst_images = ((args_aos *) p)->dst;
  int start =  ((args_aos *) p)->start_index;
  int end =  ((args_aos *) p)->end_index;

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG) {
    for (unsigned i = start; i < end; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[i + j].r / src_images[i + j].g + (F0 + F1 * F1) * 
		   src_images[i + j].b) / (F1 * src_images[i + j].b);
	v1 = v0 - F1 * src_images[i + j].b;
      }      
      dst_images[i + j].r = (src_images[i + j].r * v0 - src_images[i + j].g * F1 * v1);
      dst_images[i + j].g  = (src_images[i + j].g * F1 * (1.0f - v1) - src_images[i + j].r); 
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[i + j].b = v0;
#endif
#if defined  MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[i + j].x = v1;
#endif
    }
  }
  return NULL;
}

void *host_grayscale_aos(void *p) {
#if 0
  pixel *src_images = ((args_aos *) p)->src;
  pixel *dst_images = ((args_aos *) p)->dst;
  int start =  ((args_aos *) p)->start_index;
  int end =  ((args_aos *) p)->end_index;

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  start = start * PIXELS_PER_IMG;
  end = end * PIXELS_PER_IMG;

  for (int j = start; j < end; j += PIXELS_PER_IMG) {
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
		   src_images[i].b) / (F1 * src_images[i].b);
	v1 = v0 - F1 * src_images[i].b;
      }
      
      dst_images[i].r = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 
      dst_images[i].g  = (src_images[i].g * F1 * (1.0f - v1) - src_images[i].r);  
      dst_images[i].b = v0;
      dst_images[i].x = v1;
    }
  }
#endif
  return NULL;
}


void *host_grayscale_da_pixel(void *p) {
#if 0
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

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  DATA_ITEM_TYPE gs;
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG) 
    for (unsigned i = start; i < end; i++) { 
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = (r[i + j] / g[i + j] + (F0 + F1 * F1) * b[i + j]) / (F1 * b[i + j]);
	v1 = v0 - F1 * b[i + j];
      }
      d_r[i + j] = (r[i + j] * v0 - g[i + j] * F1 * v1);
      d_g[i + j] = (g[i + j] * F1 * (1.0f - v1) - r[i + j]); 
      d_b[i + j] = v0;
      d_x[i + j] = v1;
    }
#endif
  return NULL;
}


void *host_grayscale_da(void *p) {
#if 0
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
#endif
  return NULL;
}
