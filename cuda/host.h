typedef struct arg_aos_struct_type {
  pixel *src;
  pixel *dst;
  int start_index;
  int end_index;
} args_aos;


typedef struct arg_da_struct_type {
  float *r;
  float *g;
  float *b; 
  float *x;
  float *d_r;
  float *d_g;
  float *d_b; 
  float *d_x;
  int start_index;
  int end_index;
} args_da;


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
