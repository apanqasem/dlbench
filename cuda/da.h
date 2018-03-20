#include<kernel.h>
__global__ void grayscale(float *r, float *g) {
//, float *b, 
//			  float *dst_r, float *dst_g, float *dst_b) {

  int tidx = threadIdx.x; // + blockDim.x * blockIdx.x;

  DATA_ITEM_TYPE s0 = 0.0f;
  DATA_ITEM_TYPE s1 = 0.0f;

  g[tidx] = r[0];
}

#if 0  
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {


#if (MEM == 1)  
    KERNEL2(s1,r[tidx + j],r[tidx + j],r[tidx + j]);
#endif
#if (MEM == 2) 
    KERNEL2(s1,r[tidx + j],g[tidx + j],g[tidx + j]);
#endif
#if (MEM > 2) 
    KERNEL2(s1,r[tidx + j],b[tidx + j],g[tidx + j]);
#endif 

    for (int k = 0; k < ITERS; k++)
      KERNEL1(s0,s0,r[tidx + j]);
#if (MEM >= 1)     
    dst_r[tidx + j] = s0;
#endif
#if (MEM >= 2)     
    dst_g[tidx + j] = s1;
#endif
#if (MEM >= 3)     
    dst_images[tidx + j].b = s1;
#endif
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
	  //  }
	  //}
