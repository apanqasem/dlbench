#ifndef DLBENCH_H
#define DLBENCH_H

#define DATA_ITEM_TYPE float
#define NUM_IMGS IMGS
#define PIXELS_PER_IMG PIXELS
#define THREADS __THREADS

#define ITERS ((MEM * 4 * 2) - 2) + (INTENSITY - 1) * (MEM * 4 * 2)
//#define ITERS 0
#define FIELDS MEM

#define SWEEPS 1   
#define KERNEL_ITERS KITERS
#define CF COARSENFACTOR
#define TILE TILESIZE

#define THREADS PIXELS_PER_IMG / CF
#define WORKGROUP BLKS
#define SPARSITY SPARSITY_VAL

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


#define DEVICES 2
#define CPU_THREADS 4

#define OFFSET_R TILE * 0
#define OFFSET_G TILE * 1
#define OFFSET_B TILE * 2
#define OFFSET_X TILE * 3
#define OFFSET_A TILE * 4
#define OFFSET_C TILE * 5
#define OFFSET_D TILE * 6
#define OFFSET_E TILE * 7


#define ERROR_THRESH 0.0001f   // relaxed FP-precision checking, need for higher AI kernels

#ifdef HETERO
#define HOST
#define DEVICE
#endif

#endif // conditional definition
