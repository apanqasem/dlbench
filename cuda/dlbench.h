#ifndef DLBENCH_H
#define DLBENCH_H
#ifdef TUNE
#define ITERS __INTENSITY_FROM_TUNER
#define NUM_IMGS __NUM_IMGS_FROM_TUNER
#define PIXELS_PER_IMG __PIXELS_PER_IMG_FROM_TUNER
#define DATA_ITEM_TYPE __DATA_ITEM_TYPE_FROM_TUNER
#else 
#define DATA_ITEM_TYPE float
#define NUM_IMGS IMGS
#define PIXELS_PER_IMG PIXELS
#define THREADS __THREADS
#endif

#define ITERS 22 + (INTENSITY - 1) * 24 + ((MEM - 3) * 8) 
#define SWEEPS 1                         // floating-point ops in one iteration of kernel looPp
#define UNROLL 100
#define KERNEL_ITERS KITERS

typedef struct pixel_type {
#if (MEM >= 1)
  DATA_ITEM_TYPE r;
#endif 
#if (MEM >= 2)
  DATA_ITEM_TYPE g;
#endif
#if (MEM >= 3)
  DATA_ITEM_TYPE b;
#endif
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

typedef struct img_type {
  DATA_ITEM_TYPE *r;
  DATA_ITEM_TYPE *g;
  DATA_ITEM_TYPE *b;
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *x;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *a;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *c;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *d;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *e;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *f;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *h;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *j;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *k;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *l;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *m;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *n;
#endif
#if defined MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE *o;
#endif
#if defined MEM17 || MEM18
    DATA_ITEM_TYPE *p;
#endif
#if defined MEM18
    DATA_ITEM_TYPE *q;
#endif
} img;

#define FIELDS 3
#ifdef MEM2
#define FIELDS 2
#endif
#ifdef MEM3 
#define FIELDS 3
#endif
#ifdef MEM4 
#define FIELDS 4
#endif
#ifdef MEM5 
#define FIELDS 5
#endif
#ifdef MEM6 
#define FIELDS 6
#endif
#ifdef MEM7 
#define FIELDS 7
#endif
#ifdef MEM8 
#define FIELDS 8
#endif
#ifdef MEM9 
#define FIELDS 9
#endif
#ifdef MEM10 
#define FIELDS 10
#endif
#ifdef MEM11 
#define FIELDS 11
#endif
#ifdef MEM12 
#define FIELDS 12
#endif
#ifdef MEM13 
#define FIELDS 13
#endif
#ifdef MEM14 
#define FIELDS 14
#endif
#ifdef MEM15 
#define FIELDS 15
#endif
#ifdef MEM16 
#define FIELDS 16
#endif
#ifdef MEM17 
#define FIELDS 17
#endif
#ifdef MEM18 
#define FIELDS 18
#endif

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


#define DEVICES 2
#define CPU_THREADS 4

#define CF COARSENFACTOR
#define TILE TILESIZE

#define THREADS PIXELS_PER_IMG / CF
#define WORKGROUP 1024
#define SPARSITY SPARSITY_VAL


#define STREAMS 8

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


#define ERROR_THRESH 0.0001f   // relaxed FP-precision checking, need for higher AI kernels

#ifdef HETERO
#define HOST
#define DEVICE
#endif

#endif // conditional definition
