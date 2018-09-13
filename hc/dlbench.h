#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

#include "hc.hpp"


#define IMPLEMENTATION_STRING "HC"
#define ERROR_THRESH 0.0001f   // relaxed FP-precision checking, need for higher AI kernels

#define DATA_ITEM_TYPE double

#define SIZE IMGS
#define PIXELS_PER_IMG PIXELS
#define THREADS __THREADS
#define ITERS INTENSITY
#define WORKGROUP WKGRP

double mysecond(); 
void dlbench(DATA_ITEM_TYPE *in_a, DATA_ITEM_TYPE *in_b, DATA_ITEM_TYPE scalar, DATA_ITEM_TYPE *out, int n);
