#include <dlbench.h>



void check_results_aos(pixel *src_images, pixel *dst_images, int host_start, int device_end);

void check_results_soa(img *src_images, img *dst_images, int host_start, int device_end);

void check_results_da(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,
                      DATA_ITEM_TYPE *d_r, int host_start, int device_end);

void check_results_ca(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images, 
                      int host_start, int device_end);

