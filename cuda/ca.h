#ifdef CA
__global__ void grayscale(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images) {
 
  size_t local_id = threadIdx.x;
  size_t group_id = blockIdx.x;

  size_t tile_factor = (local_id / TILE) * TILE * (FIELDS - 1);  
  size_t thrd_offset;

  size_t loc_in_tile = (local_id * SPARSITY) % TILE;
  size_t offset_to_next_tile = (local_id / (TILE/SPARSITY)) * (FIELDS * TILE);

#if 1 //CA_OPT
  if (tile_factor == 0)
    thrd_offset = local_id + (group_id * WORKGROUP * FIELDS);
  else
    thrd_offset = local_id + tile_factor   + (group_id * WORKGROUP * FIELDS);
#else 
  int sets = (group_id / SPARSITY);    // sets processed 
  int set_offset = WORKGROUP * SPARSITY * sets;
  thrd_offset = loc_in_tile + offset_to_next_tile + (group_id - SPARSITY * sets) + (SPARSITY * FIELDS * TILE * sets);
#endif  

  size_t OR = OFFSET_R + thrd_offset;
  size_t OG = OFFSET_G + thrd_offset;
  size_t OB = OFFSET_B + thrd_offset;
  size_t OX = OFFSET_X + thrd_offset;
  size_t OA = OFFSET_A + thrd_offset;
  size_t OC = OFFSET_C + thrd_offset;
  size_t OD = OFFSET_D + thrd_offset;
  size_t OE = OFFSET_E + thrd_offset;

  for (int p = 0; p < SWEEPS; p++) {
    DATA_ITEM_TYPE alpha = 0.0f;
    for (int j = 0; j < NUM_IMGS * (FIELDS * PIXELS_PER_IMG); j = j + (PIXELS_PER_IMG * FIELDS)) {
#if (MEM == 1)  
    KERNEL2(alpha,src_images[OR + j],src_images[OR + j],src_images[OR + j]);
#endif
#if (MEM == 2) 
    KERNEL2(alpha,src_images[OR + j],src_images[OG + j],src_images[OG + j]);
#endif
#if (MEM > 2) 
    KERNEL2(alpha,src_images[OR + j],src_images[OG + j],src_images[OB + j]);
#endif 

    for (int k = 0; k < ITERS; k++)
      KERNEL1(alpha,alpha,src_images[OR + j]);

#if (MEM >= 1)     
    dst_images[OR + j] = alpha;
#endif
#if (MEM >= 2)     
    dst_images[OG + j] = alpha;
#endif
#if (MEM >= 3)     
    dst_images[OB + j] = alpha;
#endif
#if (MEM >= 4)
    dst_images[OX + j] = src_images[OX + j];
#endif
#if (MEM >= 5)
    dst_images[OX + j] = src_images[OA + j];
#endif
#if (MEM >= 6)
    dst_images[OC + j]  = src_images[OC + j];
#endif
#if (MEM >= 7)
    dst_images[OD + j] = src_images[OD + j];
#endif
#if (MEM >= 8)
    dst_images[OE + j] = src_images[OE + j];
#endif
    }
  }
}
#endif
