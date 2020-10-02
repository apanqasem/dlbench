#include<fstream>
#include<stdio.h>
#include<cstdlib>
#include<sys/time.h>
#include<pthread.h>
#include<dlbench.h>

#include<aos.h>
#include<da.h>
#include<ca.h>

#define MAX_ERRORS 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void check_results_aos(pixel *src_images, pixel *dst_images, int host_start, int device_end) {
  int errors = 0;
#ifdef HOST
  DATA_ITEM_TYPE F0 = 0.02f; 
  DATA_ITEM_TYPE F1 = 0.30f; 
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      DATA_ITEM_TYPE c0 = (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
			   src_images[i].b) / (F1 * src_images[i].b);
      DATA_ITEM_TYPE c1 = F1 * src_images[i].b; 
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + c0;
	v1 = v0 - c1;
      }
      DATA_ITEM_TYPE exp_result = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 

#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[512].r);
#endif
      DATA_ITEM_TYPE delta = fabs(dst_images[i].r - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef VERBOSE
        printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#else
  for (int i = 0; i < device_end * PIXELS_PER_IMG; i++) { 
    DATA_ITEM_TYPE alpha = 0.0f;
#if (MEM == 1)  
      KERNEL2(alpha,src_images[i].r,src_images[i].r,src_images[i].r);
#endif
#if (MEM == 2) 
      KERNEL2(alpha,src_images[i].r,src_images[i].g,src_images[i].g);
#endif
#if (MEM > 2) 
      KERNEL2(alpha,src_images[i].r,src_images[i].g,src_images[i].b);
#endif 
      for (int k = 0; k < ITERS; k++)
      	KERNEL1(alpha,alpha,src_images[i].r);

      DATA_ITEM_TYPE exp_result = alpha; 

#ifdef DEBUG 
      if (i == 512) {
	printf("%3.2f %3.2f\n", exp_result, dst_images[i].r);
      }
#endif
      DATA_ITEM_TYPE delta = fabs(dst_images[i].r - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
  }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
#endif
}

void check_results_da(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,
                      DATA_ITEM_TYPE *d_r, int host_start, int device_end) {
  int errors = 0;
#ifdef HOST
  DATA_ITEM_TYPE F0 = 0.02; 
  DATA_ITEM_TYPE F1 = 0.30; 
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG) {
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0;
      DATA_ITEM_TYPE v1 = 0.0;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
	v1 = v0 - F1 * b[i];
      }
      DATA_ITEM_TYPE exp_result = (r[i] * v0 - g[i] * F1 * v1);
#ifdef DEBUG
      if (i == 512) {
	  printf("%3.2f %3.2f\n", exp_result, d_r[i]);
      }
#endif
      DATA_ITEM_TYPE delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
	  errors++;
#ifdef VERBOSE
	  if (errors < MAX_ERRORS) 
	    printf("%d %f %f\n", i, exp_result, d_r[i]);
#endif
	}
    }
  }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#endif

#ifdef DEVICE 
  DATA_ITEM_TYPE alpha = 0.0f;
  DATA_ITEM_TYPE beta = 0.0f;
  for (int i = 0; i < device_end * PIXELS_PER_IMG; i++) {
#if (MEM == 1)  
    KERNEL2(alpha,r[i],r[i],r[i]);
#endif
#if (MEM == 2) 
    KERNEL2(alpha,r[i],g[i],g[i]);
#endif
#if (MEM > 2) 
    KERNEL2(alpha,r[i],g[i],b[i]);
#endif 

    for (int k = 0; k < ITERS; k++)
      KERNEL1(alpha,alpha,r[i]);

    DATA_ITEM_TYPE exp_result = alpha; 
#ifdef DEBUG
      if (i == 512)    // check a pixel in the middle of the image
        printf("%f %f\n", exp_result, d_r[i]);
#endif
      DATA_ITEM_TYPE delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%f %f\n", exp_result, d_r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
#endif 
  return;
}

void check_results_ca(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images, 
                      int host_start, int device_end) {

  int errors = 0;
#if 0
  DATA_ITEM_TYPE F0 = 0.02f; 
  DATA_ITEM_TYPE F1 = 0.30f; 
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
      for (int k = j; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS) {
	for (int m = k; m < k + TILE; m++) { 
	  DATA_ITEM_TYPE v0 = 0.0f;
	  DATA_ITEM_TYPE v1 = 0.0f;
	  for (int k = 0; k < ITERS; k++) {
	    v0 = v0 + (src_images[OFFSET_R + m] / src_images[OFFSET_G + m] + (F0 + F1 * F1) 
		       * src_images[OFFSET_B + m]) / (F1 * src_images[OFFSET_B + m]);
	    v1 = v0 - F1 * src_images[OFFSET_B + m];
	  }
	  DATA_ITEM_TYPE exp_result = 
	    (src_images[OFFSET_R + m] * v0 - src_images[OFFSET_G + m] * F1 * v1);
#ifdef DEBUG
	  if (m == 0)
	    printf("%f %f\n", exp_result, dst_images[OFFSET_R + m]);
#endif
	  DATA_ITEM_TYPE delta = fabs(dst_images[OFFSET_R + m] - exp_result);
	  if (delta/exp_result > ERROR_THRESH) {
	    errors++;
#ifdef DEBUG
	    if (errors < MAX_ERRORS)
	      printf("%d %f %f\n", m, exp_result, dst_images[m]);
#endif
	  }
	}
      }
#endif

    DATA_ITEM_TYPE alpha = 0.0f;
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
      for (int k = j; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS) {
	for (int m = k; m < k + TILE; m++) { 
#if (MEM == 1)  
      KERNEL2(alpha,src_images[OFFSET_R+ m],src_images[OFFSET_R+ m],src_images[OFFSET_R+ m]);
#endif
#if (MEM == 2) 
      KERNEL2(alpha,src_images[OFFSET_R+ m],src_images[OFFSET_G+ m],src_images[OFFSET_G+ m]);
#endif
#if (MEM > 2) 
      KERNEL2(alpha,src_images[OFFSET_R+ m],src_images[OFFSET_G+ m],src_images[OFFSET_B+ m]);
#endif 

      for (int k = 0; k < ITERS; k++)
        KERNEL1(alpha,alpha,src_images[OFFSET_R+ m]);
      
      DATA_ITEM_TYPE exp_result = alpha; 
#ifdef DEBUG
      if (m == (k + TILE) - 1)
	printf("%f %f %d\n", exp_result, dst_images[OFFSET_R+ m]);
#endif
      DATA_ITEM_TYPE delta = fabs(dst_images[OFFSET_R+ m] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%f %f\n", exp_result, dst_images[i]);
#endif
      }
	}
      }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


void check_results_soa(img *src_images, img *dst_images, int host_start, int device_end) {

  DATA_ITEM_TYPE F0 = 0.02f; 
  DATA_ITEM_TYPE F1 = 0.30f; 
  int errors = 0;
  for (int j = 0; j < device_end; j++) 
    for (unsigned int i = 0; i < PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[j].r[i] / src_images[j].g[i] + (F0 + F1 * F1) * 
		   src_images[j].g[i]) / (F1 * src_images[j].g[i]);
	v1 = v0 - F1 * src_images[j].g[i];
      }
      DATA_ITEM_TYPE exp_result = (src_images[j].r[i] * v0 - src_images[j].g[i] * F1 * v1);
#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[j].r[i]);
#endif
      DATA_ITEM_TYPE delta = fabs(dst_images[j].r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
          printf("%d %f %f\n", i, exp_result, dst_images[j].r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


int main(int argc,char *argv[]) {
  

#ifdef AOS
#ifdef UM
  pixel *src_images;
  gpuErrchk(cudaMallocManaged(&src_images, sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG))
#else
  pixel *src_images = (pixel *) malloc((sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG)); 
#endif

  pixel *dst_images = (pixel *) malloc(sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG); 


  /* 
   * Initialization 
   */
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
   for (int i = j; i < j + PIXELS_PER_IMG; i++) {
     src_images[i].r = 1.0; //(DATA_ITEM_TYPE)i;
#if (MEM >= 2)
     src_images[i].g = i; //i * 10.0f;
#endif
#if (MEM >= 3)     
     src_images[i].b = 1.0; //(DATA_ITEM_TYPE)i;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].x = i * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].a = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].c = i * 10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].d = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].e = i * 10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].f = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].h = i * 10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].j = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].k = i * 10.0f;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].l = (DATA_ITEM_TYPE)i;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].m = i * 10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i].n = i * 10.0f;
#endif
#if defined MEM16 || MEM17 || MEM18
	  src_images[i].o = i * 10.0f;
#endif
#if defined MEM17 || MEM18
	  src_images[i].p = i * 10.0f;
#endif
#if defined MEM18
	  src_images[i].q = i * 10.0f;
#endif
   }


  /* Copy for verification of results on CPU;  needed for C2GI pattern but doing it for other patterns as well  */
  pixel *src_images_copy = (pixel *) malloc((sizeof(pixel) * NUM_IMGS * PIXELS_PER_IMG)); 
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
   for (int i = j; i < j + PIXELS_PER_IMG; i++) {
     src_images_copy[i].r = src_images[i].r;
#if (MEM >= 2)
     src_images_copy[i].g = src_images[i].g; 
#endif
#if (MEM >= 3)
     src_images_copy[i].b = src_images[i].b;
#endif
#if (MEM >= 4)
     src_images_copy[i].x = src_images[i].x;
#endif
   }


  pixel *d_src_images;
  pixel *d_dst_images;
#endif

#ifdef DA
#ifdef UM
  DATA_ITEM_TYPE *r, *g, *b, *x, *a, *c, *d, *e; 
  gpuErrchk(cudaMallocManaged(&r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS)); 
  gpuErrchk(cudaMallocManaged(&e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#else
  DATA_ITEM_TYPE *r = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *g = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *b = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *x = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *a = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *c = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *d = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *e = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
#endif

  DATA_ITEM_TYPE *dst_r = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_g = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_b = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_x = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_a = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_c = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_d = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 
  DATA_ITEM_TYPE *dst_e = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS); 

  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
    for (int i = j; i < j + PIXELS_PER_IMG; i++) {
      r[i] = (DATA_ITEM_TYPE)i;
      g[i] = i * 10.0f;
      b[i] = (DATA_ITEM_TYPE)i;
      x[i] = i * 10.0f;
      a[i] = (DATA_ITEM_TYPE)i;
      c[i] = i * 10.0f;
      d[i] = (DATA_ITEM_TYPE)i;
      e[i] = i * 10.0f;
  }

  DATA_ITEM_TYPE *d_r, *d_g, *d_b, *d_x, *d_a, *d_c, *d_d, *d_e; 
  DATA_ITEM_TYPE *d_dst_r, *d_dst_g, *d_dst_b, *d_dst_x, *d_dst_a, *d_dst_c, *d_dst_d, *d_dst_e; 
#endif

#ifdef CA
  unsigned long size = NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
  DATA_ITEM_TYPE *src_images;
  DATA_ITEM_TYPE *dst_images;

  src_images = (DATA_ITEM_TYPE *) malloc(size);
  dst_images = (DATA_ITEM_TYPE *) malloc(size);

  if (!src_images) {
    printf("Unable to malloc src_images to fine grain memory. Exiting\n");
    exit(0);
  }
  if (!dst_images) {
    printf("Unable to malloc dst_images to fine grain memory. Exiting\n");
    exit(0);
  }

  int loc_sparsity = SPARSITY;
  int this_set_tile_count = 0;
  for (int j = 0; j < NUM_IMGS * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
    for (int k = j, t = 0; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS, t++) {
	for (int m = k, n = 0; m < k + TILE; m++, n++) { 	
	  if (t == loc_sparsity) {
	    this_set_tile_count++;
	    t = 0;
	  }
	  int val = (n * loc_sparsity + t) + (this_set_tile_count * loc_sparsity * TILE);
	  src_images[OFFSET_R + m] = (DATA_ITEM_TYPE)val;
	  src_images[OFFSET_G + m] = val *10.0f;
#if (MEM >= 3)
	  src_images[OFFSET_B + m] = (DATA_ITEM_TYPE)val;
#endif
#if (MEM >= 4)
	  src_images[OFFSET_X + m] = val * 10.0f;
#endif
#if (MEM >= 5)
      src_images[OFFSET_A + m] = (DATA_ITEM_TYPE)val;
#endif
#if (MEM >= 6)
      src_images[OFFSET_C + m] = val * 10.0f;
#endif
#if (MEM >= 7)
      src_images[OFFSET_D + m] = (DATA_ITEM_TYPE)val;
#endif
 #if (MEM >= 8)
       src_images[OFFSET_E + m] = val * 10.0f;
 #endif
	 }
     }

   DATA_ITEM_TYPE *d_src_images;
   DATA_ITEM_TYPE *d_dst_images;
 #endif

 #ifdef SOA
   img *src_images;
   img *dst_images;

   src_images = (img *) malloc(sizeof(img) * NUM_IMGS);
   dst_images = (img *) malloc(sizeof(img) * NUM_IMGS);

   if (!src_images) {
     printf("Unable to malloc src_images to fine grain memory. Exiting\n");
     exit(0);
   }
   if (!dst_images) {
     printf("Unable to malloc dst_images to fine grain memory. Exiting\n");
     exit(0);
   }
   for (int j = 0; j < NUM_IMGS; j++) {
     src_images[j].r = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG); 
     src_images[j].g = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG); 
     src_images[j].b = (DATA_ITEM_TYPE *) malloc(sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG); 
   }

   for (int j = 0; j < NUM_IMGS; j++) 
     for (int k = 0; k < PIXELS_PER_IMG; k++) {
       src_images[j].r[k] = (DATA_ITEM_TYPE)k;
       src_images[j].g[k] = k * 10.0f;
       src_images[j].b[k] = (DATA_ITEM_TYPE)k;
       //      src_images[j].x[k] = k * 10.0f;
     }

   img *d_src_images;
   img *d_dst_images;
 #endif 

   int host_start = 0;
   int device_end = NUM_IMGS;

 #ifdef HETERO 
   unsigned int imgs_per_device = NUM_IMGS/DEVICES;
   device_end = host_start = imgs_per_device;   // device [0..(N/2)], host [(N/2 + 1) ... N]
 #endif 

   // layout change: copy src data into DA 
 #ifdef COPY
   copy_aos_to_da(r, g, b, x, src_images);
 #endif

   double t;
 #ifdef HOST
   pthread_t threads[CPU_THREADS];
   unsigned int num_imgs_per_thread = ((NUM_IMGS - host_start)/CPU_THREADS);
   int start = host_start;

   t = mysecond();
 #ifdef AOS
   args_aos host_args[CPU_THREADS];
 #else
   args_da host_args[CPU_THREADS];
 #endif
   for (int i = 0; i < CPU_THREADS; i++) {
     host_args[i].start_index = start;
     host_args[i].end_index = start + num_imgs_per_thread;
 #ifdef AOS
     // entire array passed to each thread; should consider passing sections
     host_args[i].src = src_images;
     host_args[i].dst = dst_images;
     pthread_create(&threads[i], NULL, &host_grayscale_aos, (void *) &host_args[i]);
 #else 
     host_args[i].r = r;
     host_args[i].g = g;
     host_args[i].b = b;
     host_args[i].x = x;
     host_args[i].d_r = dst_r;
     host_args[i].d_g = dst_g;
     host_args[i].d_b = dst_b;
     host_args[i].d_x = dst_x;
     pthread_create(&threads[i], NULL, &host_grayscale_da, (void *) &host_args[i]);
 #endif 
     start = start + num_imgs_per_thread;
   }
 #endif

 #ifdef DEVICE 
   cudaEvent_t start_copy_to_dev , stop_copy_to_dev;
   cudaEvent_t start_copy_to_host , stop_copy_to_host;
   float msec_with_copy_to_dev = 0.0f;
   float msec_with_copy_to_host = 0.0f;


   gpuErrchk(cudaEventCreate(&start_copy_to_dev));
   gpuErrchk(cudaEventCreate(&stop_copy_to_dev));
   gpuErrchk(cudaEventCreate(&start_copy_to_host));
   gpuErrchk(cudaEventCreate(&stop_copy_to_host));


 #ifdef AOS
 #ifndef UM
   gpuErrchk(cudaMalloc((void **) &d_src_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel)));
 #endif
   gpuErrchk(cudaMalloc((void **) &d_dst_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel)));

 #ifndef UM
    gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
    gpuErrchk(cudaMemcpy(d_src_images,src_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel),cudaMemcpyHostToDevice));
    gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
 #endif
 #endif
#ifdef DA
#ifndef UM
    gpuErrchk(cudaMalloc((void **) &d_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#if (MEM >= 2)
   gpuErrchk(cudaMalloc((void **) &d_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 3)
   gpuErrchk(cudaMalloc((void **) &d_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 4)
   gpuErrchk(cudaMalloc((void **) &d_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 5)
   gpuErrchk(cudaMalloc((void **) &d_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 6)
   gpuErrchk(cudaMalloc((void **) &d_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 7)
   gpuErrchk(cudaMalloc((void **) &d_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 8)
   gpuErrchk(cudaMalloc((void **) &d_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#endif
   gpuErrchk(cudaMalloc((void **) &d_dst_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#if (MEM >= 2)
   gpuErrchk(cudaMalloc((void **) &d_dst_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 3)
   gpuErrchk(cudaMalloc((void **) &d_dst_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 4)
   gpuErrchk(cudaMalloc((void **) &d_dst_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 5)
   gpuErrchk(cudaMalloc((void **) &d_dst_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 6)
   gpuErrchk(cudaMalloc((void **) &d_dst_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 7)
   gpuErrchk(cudaMalloc((void **) &d_dst_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif
#if (MEM >= 8)
   gpuErrchk(cudaMalloc((void **) &d_dst_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS));
#endif

#ifndef UM
   gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
   gpuErrchk(cudaMemcpy(d_r, r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#if (MEM >= 2)
   gpuErrchk(cudaMemcpy(d_g, g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 3)
   gpuErrchk(cudaMemcpy(d_b, b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 4)
   gpuErrchk(cudaMemcpy(d_x, x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 5)
   gpuErrchk(cudaMemcpy(d_a, a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 6)
   gpuErrchk(cudaMemcpy(d_c, c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 7)
   gpuErrchk(cudaMemcpy(d_d, d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
#if (MEM >= 8)
   gpuErrchk(cudaMemcpy(d_e, e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyHostToDevice));
#endif
   gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
#endif
#endif
 #ifdef CA
   gpuErrchk(cudaMalloc((void **) &d_src_images, size));
   gpuErrchk(cudaMalloc((void **) &d_dst_images, size)); 
   gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
   gpuErrchk(cudaMemcpy(d_src_images,src_images, size, cudaMemcpyHostToDevice));
   gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
 #endif 
 #ifdef SOA
   gpuErrchk(cudaMalloc((void **) &d_src_images, sizeof(img) * NUM_IMGS));
   gpuErrchk(cudaMalloc((void **) &d_dst_images, sizeof(img) * NUM_IMGS)); 

   gpuErrchk(cudaEventRecord(start_copy_to_dev,0));
   gpuErrchk(cudaMemcpy(d_dst_images,dst_images, sizeof(img) * NUM_IMGS, cudaMemcpyHostToDevice));
   gpuErrchk(cudaMemcpy(d_src_images,src_images, sizeof(img) * NUM_IMGS, cudaMemcpyHostToDevice))
   gpuErrchk(cudaEventRecord(stop_copy_to_dev,0));
 #endif
   cudaEvent_t start_kernel , stop_kernel;
   float msecTotal = 0.0f;
   gpuErrchk(cudaEventCreate(&start_kernel));
   gpuErrchk(cudaEventCreate(&stop_kernel));

   gpuErrchk(cudaEventRecord(start_kernel,0));

  int threadsPerBlock = WORKGROUP;
  int blocksPerGrid = THREADS/WORKGROUP; 
 #ifdef C2GI
 #ifdef AOS
  for (int i = 0; i < KERNEL_ITERS; i++) {
 #ifdef UM
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(src_images, d_dst_images);
    cudaDeviceSynchronize();
 #else
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);
 #endif
  }
 #endif
 #endif

 #ifdef C2G
 #ifdef AOS

 #ifdef UM
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(src_images, d_dst_images);
 #else
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);
 #endif

 #endif
 #endif

#ifdef DA
#ifdef UM 
#if (MEM == 1)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r,d_dst_r);
 #endif
 #if (MEM == 2)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, 
						 d_dst_r, d_dst_g);
 #endif
 #if (MEM == 3)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, 			  
						 d_dst_r, d_dst_g, d_dst_b);
 #endif
 #if (MEM == 4)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, x,  			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x);
 #endif
 #if (MEM == 5)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, x, a, 			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a);
 #endif
 #if (MEM == 6)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, x, a, c, 			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, d_dst_c);
 #endif
 #if (MEM == 7)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, x, a, c, d, 	
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, 
						 d_dst_c, d_dst_d);
 #endif
 #if (MEM == 8)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(r, g, b, x, a, c, d, e, 
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, 
						 d_dst_c, d_dst_d, d_dst_e);
 #endif
#else
#if (MEM == 1)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r,d_dst_r);
 #endif
 #if (MEM == 2)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, 
						 d_dst_r, d_dst_g);
 #endif
 #if (MEM == 3)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, 			  
						 d_dst_r, d_dst_g, d_dst_b);
 #endif
 #if (MEM == 4)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x,  			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x);
 #endif
 #if (MEM == 5)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x, d_a, 			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a);
 #endif
 #if (MEM == 6)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x, d_a, d_c, 			  
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, d_dst_c);
 #endif
 #if (MEM == 7)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x, d_a, d_c, d_d, 	
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, 
						 d_dst_c, d_dst_d);
 #endif
 #if (MEM == 8)
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_r, d_g, d_b, d_x, d_a, d_c, d_d, d_e, 
						 d_dst_r, d_dst_g, d_dst_b, d_dst_x, d_dst_a, 
						 d_dst_c, d_dst_d, d_dst_e);
 #endif
#endif
 #endif
 #ifdef CA
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);  
 #endif
 #ifdef SOA
    grayscale<<<blocksPerGrid,threadsPerBlock>>>(d_src_images, d_dst_images);  
 #endif
    gpuErrchk(cudaEventRecord(stop_kernel,0));

    gpuErrchk(cudaEventSynchronize(start_kernel));
    gpuErrchk(cudaEventSynchronize(stop_kernel));
    gpuErrchk(cudaEventElapsedTime(&msecTotal, start_kernel, stop_kernel));
    
#ifndef UM
   gpuErrchk(cudaEventSynchronize(start_copy_to_host));
   gpuErrchk(cudaEventSynchronize(stop_copy_to_host));
   gpuErrchk(cudaEventElapsedTime(&msec_with_copy_to_dev, start_copy_to_dev, stop_copy_to_dev));
#endif
   
#ifdef AOS
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, PIXELS_PER_IMG * NUM_IMGS * sizeof(pixel),cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef DA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_r, d_dst_r, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#if (MEM >= 2)
  gpuErrchk(cudaMemcpy(dst_g, d_dst_g, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 3)
  gpuErrchk(cudaMemcpy(dst_b, d_dst_b, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 4)
  gpuErrchk(cudaMemcpy(dst_x, d_dst_x, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 5)
  gpuErrchk(cudaMemcpy(dst_a, d_dst_a, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 6)
  gpuErrchk(cudaMemcpy(dst_c, d_dst_c, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 7)
  gpuErrchk(cudaMemcpy(dst_d, d_dst_d, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
#if (MEM >= 8)
  gpuErrchk(cudaMemcpy(dst_e, d_dst_e, sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS,cudaMemcpyDeviceToHost)); 
#endif
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef CA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, size, cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#ifdef SOA
  gpuErrchk(cudaEventRecord(start_copy_to_host,0));
  gpuErrchk(cudaMemcpy(dst_images, d_dst_images, sizeof(img) * NUM_IMGS, cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaEventRecord(stop_copy_to_host,0));
#endif
#endif

#ifdef HOST
  for (int i = 0; i < CPU_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  t = 1.0E6 * (mysecond() - t);
#endif


#ifdef COPY 
  check_results_aos(src_images, dst_images, host_start, device_end);
  //  check_results_da(r, g, b, x, d_r, host_start, device_end);
#else
#ifdef AOS
  check_results_aos(src_images, dst_images, host_start, device_end);
#endif
#ifdef DA
  //  check_results_da(r, g, b, x, dst_r, host_start, device_end);
#endif
#ifdef CA
  check_results_ca(src_images, dst_images, host_start, device_end);
#endif
#ifdef SOA
  check_results_soa(src_images, dst_images, host_start, device_end);
#endif
#endif

  /* derive perf metrics */
  unsigned long dataMB = (NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS)/(1024 * 1024);
  double flop = (6.0  * (DATA_ITEM_TYPE) ITERS * (DATA_ITEM_TYPE) NUM_IMGS * (DATA_ITEM_TYPE) PIXELS_PER_IMG);

#ifndef HETERO
#ifdef DEVICE 
  t = msecTotal * 1000;
  double t_with_copy = (msec_with_copy_to_dev + msec_with_copy_to_host) * 1000;
  double secs_with_copy = t_with_copy/1000000;
#endif
#endif

  double secs = t/1000000;

#ifdef VERBOSE
  fprintf(stdout, "Kernel execution time %3.2f ms\n", t/1000);
  fprintf(stdout, "Attained bandwidth: %3.2f MB/s\n", dataMB/secs); 
  fprintf(stdout, "MFLOPS: %3.2f\n", (flop/secs)/(1024 * 1024)); 
#ifndef HETERO
#ifdef DEVICE
  fprintf(stdout, "Kernel execution time with copy %3.2f ms\n", t_with_copy/1000);
  fprintf(stdout, "Attained bandwidth with copy: %3.2f MB/s\n", dataMB/secs_with_copy); 
  fprintf(stdout, "MFLOPS with copy: %3.2f\n", (flop/secs_with_copy)/(1024 * 1024)); 
#endif
#endif
#else 
#ifdef HOST
  fprintf(stdout, "%3.2f ", t/1000);
  fprintf(stdout, "%3.2f ", dataMB/secs); 
  fprintf(stdout, "%3.2f\n ", (flop/secs)/(1024 * 1024)); 
#endif
#ifndef HETERO
#ifdef DEVICE
  fprintf(stdout, "%3.2f,", t/1000);
  fprintf(stdout, "%3.2f\n", (t_with_copy/1000));
#endif
#endif
#endif

  // move device reset at the very end. if host touches cudaMallocManaged() codes 
  gpuErrchk(cudaDeviceReset());

  return 0;
}

