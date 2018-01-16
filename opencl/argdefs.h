#ifndef ARGDEFS_H
#define ARGDEFS_H
#include <stdint.h>
typedef struct __attribute__ ((aligned(16))) brig_aos_args_type {
  uint64_t global_offset_0;
  uint64_t global_offset_1;
  uint64_t global_offset_2;
  uint64_t printf_buffer;
  uint64_t vqueue_pointer;
  uint64_t aqlwrap_pointer;
  void* in;
  void* out;
  int num_imgs;
} brig_aos_arg;

typedef struct __attribute__ ((aligned(16))) brig_da_arg_type {
  uint64_t global_offset_0;
  uint64_t global_offset_1;
  uint64_t global_offset_2;
  uint64_t printf_buffer;
  uint64_t vqueue_pointer;
  uint64_t aqlwrap_pointer;
  void* r;
  void* g;
  void* b;
  void* x;
  void* a;
  void* c;
  void* d;
  void* e;
  void* f;
  void* h;
  void* j;
  void* k;
  void* l;
  void* m;
  void* n;
  void* o;
  void* p;
  void* q;
  void* d_r;
  void* d_g;
  void* d_b;
  void* d_x;
  void* d_a;
  void* d_c;
  void* d_d;
  void* d_e;
  void* d_f;
  void* d_h;
  void* d_j;
  void* d_k;
  void* d_l;
  void* d_m;
  void* d_n;
  void* d_o;
  void* d_p;
  void* d_q;
  int num_imgs;
} brig_da_arg;

typedef struct __attribute__ ((aligned(16))) brig_soa_arg_type {
  uint64_t global_offset_0;
  uint64_t global_offset_1;
  uint64_t global_offset_2;
  uint64_t printf_buffer;
  uint64_t vqueue_pointer;
  uint64_t aqlwrap_pointer;
  void* in;
  void* out;
  int num_imgs;
} brig_soa_arg;

typedef struct __attribute__ ((aligned(16))) brig_ca_arg_type {
  uint64_t global_offset_0;
  uint64_t global_offset_1;
  uint64_t global_offset_2;
  uint64_t printf_buffer;
  uint64_t vqueue_pointer;
  uint64_t aqlwrap_pointer;
  void* src_images;
  void* dst_images;
  int num_imgs;
} brig_ca_arg; 

typedef struct __attribute__ ((aligned(16))) gcn_generic_arg_type {
  void* in;
  void* out;
  int num_imgs;
} gcn_generic_arg; 

typedef struct __attribute__ ((aligned(16))) gcn_da_arg_type {
  void* r;
  void* g;
  void* b;
  void* d_r;
  void* d_g;
  void* d_b;
#if 0
  void* x;
  void* a;
  void* c;
  void* d;
  void* e;
  void* f;
  void* h;
  void* j;
  void* k;
  void* l;
  void* m;
  void* n;
  void* o;
  void* p;
  void* q;
  void* d_x;
  void* d_a;
  void* d_c;
  void* d_d;
  void* d_e;
  void* d_f;
  void* d_h;
  void* d_j;
  void* d_k;
  void* d_l;
  void* d_m;
  void* d_n;
  void* d_o;
  void* d_p;
  void* d_q;
#endif
  int num_imgs;
} gcn_da_arg;
#endif

