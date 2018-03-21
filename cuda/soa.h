#ifdef SOA
__global__ void grayscale(img *src_images, img *dst_images) {
   int tidx = threadIdx.x + blockDim.x * blockIdx.x;
   float F0 = 0.02f; 
   float F1 = 0.30f; 
   
  for (int j = 0; j < NUM_IMGS; j++) {
    DATA_ITEM_TYPE v0 = 0.0f;
    DATA_ITEM_TYPE v1 = 0.0f;
    DATA_ITEM_TYPE c0 = (src_images[j].r[tidx] / src_images[j].g[tidx] + (F0 + F1 * F1) * 
			 src_images[j].b[tidx]) / (F1 * src_images[j].b[tidx]);
    DATA_ITEM_TYPE c1 = F1 * src_images[j].b[tidx];

    //#pragma unroll UNROLL
    for (int k = 0; k < ITERS; k++) {
      v0 = v0 + c0;
      v1 = v0 - c1;
    }
    dst_images[j].r[tidx] = (src_images[j].r[tidx] * v0 - src_images[j].g[tidx] * F1 * v1);
    dst_images[j].g[tidx]  = (src_images[j].g[tidx] * F1 * (1.0f - v1) - src_images[j].r[tidx]); 
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].b[tidx] = v0;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].x[tidx] = v1;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].a[tidx] = v0; 
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].c[tidx] = v1; 
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].d[tidx] = v0;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].e[tidx] = v1;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].f[tidx] = v0; 
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].h[tidx] = v1; 
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].j[tidx] = v0; 
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].k[tidx] = v1; 
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].l[tidx] = v0; 
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].m[tidx] = v1; 
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dst_images[j].n[tidx] = v1; 
#endif
#if defined MEM16 || MEM17 || MEM18
      dst_images[j].o[tidx] = v1; 
#endif
#if defined MEM17 || MEM18
      dst_images[j].p[tidx] = v1; 
#endif
#if defined MEM18
      dst_images[j].q[tidx] = v1; 
#endif
  }

}
#endif

