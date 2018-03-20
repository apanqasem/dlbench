#include<kernel.h>
#ifdef AOS
__global__ void grayscale(pixel *src_images, pixel *dst_images) {

  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
#if 0
  int tidx = threadIdx.x; // + blockDim.x * blockIdx.x;

  int sets = (blockIdx.x / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  tidx = (tidx * SPARSITY) + (blockIdx.x - SPARSITY * sets) + set_offset;
#endif

  DATA_ITEM_TYPE s0 = 0.0f;
  DATA_ITEM_TYPE s1 = 0.0f;
  
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {

#if (MEM == 1)  
    KERNEL2(s1,src_images[tidx+ j].r,src_images[tidx + j].r,src_images[tidx + j].r);
#endif
#if (MEM == 2) 
    KERNEL2(s1,src_images[tidx+ j].r,src_images[tidx + j].g,src_images[tidx + j].g);
#endif
#if (MEM > 2) 
    KERNEL2(s1,src_images[tidx+ j].r,src_images[tidx + j].b,src_images[tidx + j].g);
#endif 

    for (int k = 0; k < ITERS; k++)
      KERNEL1(s0,s0,src_images[tidx + j].r);
#if (MEM >= 1)     
    dst_images[tidx + j].r = s0;
#endif
#if (MEM >= 2)     
    dst_images[tidx + j].g = s1;
#endif
#if (MEM >= 3)     
    dst_images[tidx + j].b = s1;
#endif

#if 0
#if (MEM >= 4)
	  dst_images[tidx + j].x = s1;
#endif
#if (MEM >= 5)
	  dst_images[tidx + j].a = s0;
#endif
#if (MEM >= 6)
	  dst_images[tidx + j].c  = s1;
#endif
#if (MEM >= 7)
	  dst_images[tidx + j].d = s0;
#endif
#if (MEM >= 8)
	  dst_images[tidx + j].e = s1;
#endif
#if (MEM >= 9)
	  dst_images[tidx + j].f = s0;
#endif
#if (MEM >= 10)
	  dst_images[tidx + j].h  = s1;
#endif
#if (MEM >= 11)
	  dst_images[tidx + j].j = s0;
#endif
#if (MEM >= 12)
	  dst_images[tidx + j].k  = s1;
#endif
#if (MEM >= 13)
	  dst_images[tidx + j].l = s0;
#endif
#if (MEM >= 14)
	  dst_images[tidx + j].m  = s1;
#endif
#if (MEM >= 15)
	  dst_images[tidx + j].n  = s1;
#endif
#if (MEM >= 16)
	  dst_images[tidx + j].o  = s1;
#endif
#if (MEM >= 17)
	  dst_images[tidx + j].p  = s1;
#endif
#if (MEM >= 18)
	  dst_images[tidx + j].q  = s1;
#endif
#endif 
  }
}
#endif
