#include<kernel.h>

#ifdef AOS
__global__ void grayscale(pixel *src_images, pixel *dst_images) {

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
    KERNEL2(alpha,src_images[tidx + j].r,src_images[tidx + j].r,src_images[tidx + j].r);
#endif
#if (MEM == 2) 
    KERNEL2(alpha,src_images[tidx + j].r,src_images[tidx + j].g,src_images[tidx + j].g);
#endif
#if (MEM > 2) 
    KERNEL2(alpha,src_images[tidx + j].r,src_images[tidx + j].g,src_images[tidx + j].b);
#endif 

    for (int k = 0; k < ITERS; k++)
      KERNEL1(alpha,alpha,src_images[tidx + j].r);

#if (MEM >= 1)     
    dst_images[tidx + j].r = alpha;
#endif
#if (MEM >= 2)     
    dst_images[tidx + j].g = alpha;
#endif
#if (MEM >= 3)     
    dst_images[tidx + j].b = alpha;
#endif
#if (MEM >= 4)
    dst_images[tidx + j].x = src_images[tidx + j].x;
#endif
#if (MEM >= 5)
    dst_images[tidx + j].a = src_images[tidx + j].a;
#endif
#if (MEM >= 6)
    dst_images[tidx + j].c  = src_images[tidx + j].c;
#endif
#if (MEM >= 7)
    dst_images[tidx + j].d = src_images[tidx + j].d;
#endif
#if (MEM >= 8)
    dst_images[tidx + j].e = src_images[tidx + j].e;
#endif
  }
}
#endif
