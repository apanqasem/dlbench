#ifndef DLBENCH_H
#define DLBENCH_H


#define DATA_ITEM_TYPE double

#define NUM_IMGS IMGS
#define PIXELS_PER_IMG PIXELS

#define ITERS INTENSITY
#define SPARSITY SPARSITY_VAL 

#define SWEEPS SWEEP_VAL    

#define CF COARSENFACTOR
//#define THREADS PIXELS_PER_IMG / CF 

#define THREADS __THREADS
#define WORKGROUP 64
#define UNROLL 1
#define KERNEL_ITERS KITERS

// CA specific parameters 
#define TILE TILESIZE

#define OFFSET_R TILE * 0
#define OFFSET_G TILE * 1
#define OFFSET_B TILE * 2
#define OFFSET_X TILE * 3
#define OFFSET_A TILE * 4
#define OFFSET_C TILE * 5
#define OFFSET_D TILE * 6
#define OFFSET_E TILE * 7
#define OFFSET_F TILE * 8
#define OFFSET_H TILE * 9
#define OFFSET_J TILE * 10
#define OFFSET_K TILE * 11
#define OFFSET_L TILE * 12
#define OFFSET_M TILE * 13 
#define OFFSET_N TILE * 14
#define OFFSET_O TILE * 15
#define OFFSET_P TILE * 16
#define OFFSET_Q TILE * 17 


#define DEVICES 2
#define CPU_THREADS 8
#define ERROR_THRESH 0.0001f   // relaxed FP-precision checking, need for higher AI kernels


#ifdef HETERO
#define HOST
#define DEVICE
#endif

typedef struct pixel_type {
  DATA_ITEM_TYPE r;
  DATA_ITEM_TYPE g;
  DATA_ITEM_TYPE b;
#if (MEM >= 4)
  DATA_ITEM_TYPE x;
#endif
#if (MEM >= 5)
    DATA_ITEM_TYPE a;
#endif
#if (MEM >= 6)
    DATA_ITEM_TYPE c;
#endif
#if (MEM >= 7)
    DATA_ITEM_TYPE d;
#endif
#if (MEM >= 8)
    DATA_ITEM_TYPE e;
#endif
#if (MEM >= 9)
    DATA_ITEM_TYPE f;
#endif
#if (MEM >= 10)
    DATA_ITEM_TYPE h;
#endif
#if (MEM >= 11)
    DATA_ITEM_TYPE j;
#endif
#if (MEM >= 12)
    DATA_ITEM_TYPE k;
#endif
#if (MEM >= 13)
    DATA_ITEM_TYPE l;
#endif
#if (MEM >= 14)
    DATA_ITEM_TYPE m;
#endif
#if (MEM >= 15)
    DATA_ITEM_TYPE n;
#endif
#if (MEM >= 16)
    DATA_ITEM_TYPE o;
#endif
#if (MEM >= 17)
    DATA_ITEM_TYPE p;
#endif
#if (MEM >= 18)
    DATA_ITEM_TYPE q;
#endif
  } pixel;


#if (MEM == 2)
#define FIELDS 3
#else 
#define FIELDS MEM
#endif

typedef struct img_type {
  DATA_ITEM_TYPE *r;
  DATA_ITEM_TYPE *g;
  DATA_ITEM_TYPE *b;
#if (MEM >= 4) 
    DATA_ITEM_TYPE *x;
#endif
#if (MEM >= 5) 
    DATA_ITEM_TYPE *a;
#endif
#if (MEM >= 6) 
    DATA_ITEM_TYPE *c;
#endif
#if (MEM >= 7) 
    DATA_ITEM_TYPE *d;
#endif
#if (MEM >= 8) 
    DATA_ITEM_TYPE *e;
#endif
#if (MEM >= 9) 
    DATA_ITEM_TYPE *f;
#endif
#if (MEM >= 10) 
    DATA_ITEM_TYPE *h;
#endif
#if (MEM >= 11) 
    DATA_ITEM_TYPE *j;
#endif
#if (MEM >= 12) 
    DATA_ITEM_TYPE *k;
#endif
#if (MEM >= 13) 
    DATA_ITEM_TYPE *l;
#endif
#if (MEM >= 14) 
    DATA_ITEM_TYPE *m;
#endif
} img;

typedef struct arg_aos_struct_type {
  pixel *src;
  pixel *dst;
  int start_index;
  int end_index;
} args_aos;


typedef struct arg_da_struct_type {
  float *r;
  float *d_r;
  int start_index;
  int end_index;
} args_da;



#endif // conditional definition
