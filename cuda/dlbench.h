#ifndef DLBENCH_H
#define DLBENCH_H
#ifdef TUNE
#define ITERS __INTENSITY_FROM_TUNER
#define NUM_IMGS __NUM_IMGS_FROM_TUNER
#define PIXELS_PER_IMG __PIXELS_PER_IMG_FROM_TUNER
#define DATA_ITEM_TYPE __DATA_ITEM_TYPE_FROM_TUNER
#else 
#define DATA_ITEM_TYPE float
#define ITERS INTENSITY
#define NUM_IMGS IMGS
#define PIXELS_PER_IMG PIXELS
#define THREADS __THREADS
#endif

#define SWEEPS 1                         // floating-point ops in one iteration of kernel looPp
#define UNROLL 100
#define KERNEL_ITERS KITERS

typedef struct pixel_type {
  DATA_ITEM_TYPE r;
  DATA_ITEM_TYPE g;
  DATA_ITEM_TYPE b;
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE x;
  #endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE a;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE c;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE d;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE e;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE f;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE h;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE j;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE k;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE l;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE m;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE n;
#endif
#if defined MEM16 || MEM17 || MEM18
    DATA_ITEM_TYPE o;
#endif
#if defined MEM17 || MEM18
    DATA_ITEM_TYPE p;
#endif
#if defined MEM18
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

#ifdef MEM2
#define FIELDS 3
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
#define WORKGROUP 64
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
