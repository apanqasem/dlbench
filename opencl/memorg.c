#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mapi.h>
#include <memorg.h>
#include <dlbench.h>
#include <util.h>

#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed.\n", #msg); \
    exit(1); \
} 

/* Count available CPU and GPU agents */
int check_agent_info(int *gpu_agents, int *cpu_agents) {

  hsa_status_t err;
  unsigned num_agents = 0;
  err = hsa_iterate_agents(count_agents, &num_agents);
  check(Getting total number of agents, err);

#ifdef VERBOSE
  printf("Number of available agents: %d\n", num_agents);
#endif
  unsigned num_gpu_agents = 0;
  err = hsa_iterate_agents(count_gpu_agents, &num_gpu_agents);
  check(Getting number of GPU agents, err);
  
  if (num_gpu_agents < 1) {
    printf("No GPU agents found. Exiting");
    return 0;
  }
  if ((*gpu_agents) > num_gpu_agents) {
    printf("Too many GPU agents requested, setting to max available: %d.\n", num_gpu_agents);
    (*gpu_agents) = num_gpu_agents;
  }
  unsigned num_cpu_agents = num_agents - num_gpu_agents;
  
  if (num_cpu_agents < 1) { // should never happen really 
    printf("No CPU agents found. Exiting");
    return 0;
  }
  if ((*cpu_agents) > num_cpu_agents) {
    printf("Too many CPU agents requested, setting to max available: %d.\n", num_cpu_agents);
    (*cpu_agents) = num_cpu_agents;
  }
#ifdef VERBOSE
  printf("Available GPU agents: %d\n", num_gpu_agents);
  printf("Available CPU agents: %d\n", num_cpu_agents);
#endif    
  return num_agents;
}


void get_agent_handles(hsa_agent_t *gpu_agents, hsa_agent_t *cpu_agents) {

  /* STUB */
  return;
}

/********************************** 
 * 
 *    Argument assignment: BRIG 
 *  
 **********************************/
void assign_brig_args_aos(brig_aos_arg *args, pixel **src_images, pixel **dst_images, 
			  pixel **dev_src_images, pixel **dev_dst_images, 
			  int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
    args[i].in = dev_src_images[i];
    args[i].out = dev_dst_images[i];
#else
    args[i].in = src_images[i];
    args[i].out = dst_images[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  // reset this for rest of the program
  objs_per_device =  objs / gpu_agents_used;
  return; 
}

void assign_brig_args_da(brig_da_arg *args, 
			 DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
			 DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
			 DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
			 DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
			 DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
			 DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
			 DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
			 DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
			 DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
			 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
			 DATA_ITEM_TYPE **d_b, DATA_ITEM_TYPE **d_x, 
			 DATA_ITEM_TYPE **d_a, DATA_ITEM_TYPE **d_c, 
			 DATA_ITEM_TYPE **d_d, DATA_ITEM_TYPE **d_e, 
			 DATA_ITEM_TYPE **d_f, DATA_ITEM_TYPE **d_h, 
			 DATA_ITEM_TYPE **d_j, DATA_ITEM_TYPE **d_k, 
			 DATA_ITEM_TYPE **d_l, DATA_ITEM_TYPE **d_m, 
			 DATA_ITEM_TYPE **d_n, DATA_ITEM_TYPE **d_o, 
			 DATA_ITEM_TYPE **d_p, DATA_ITEM_TYPE **d_q, 
			 DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_g, 
			 DATA_ITEM_TYPE **dev_b, DATA_ITEM_TYPE **dev_x, 
			 DATA_ITEM_TYPE **dev_a, DATA_ITEM_TYPE **dev_c, 
			 DATA_ITEM_TYPE **dev_d, DATA_ITEM_TYPE **dev_e, 
			 DATA_ITEM_TYPE **dev_f, DATA_ITEM_TYPE **dev_h, 
			 DATA_ITEM_TYPE **dev_j, DATA_ITEM_TYPE **dev_k, 
			 DATA_ITEM_TYPE **dev_l, DATA_ITEM_TYPE **dev_m, 
			 DATA_ITEM_TYPE **dev_n, DATA_ITEM_TYPE **dev_o, 
			 DATA_ITEM_TYPE **dev_p, DATA_ITEM_TYPE **dev_q, 
			 DATA_ITEM_TYPE **dev_d_r, DATA_ITEM_TYPE **dev_d_g, 
			 DATA_ITEM_TYPE **dev_d_b, DATA_ITEM_TYPE **dev_d_x, 
			 DATA_ITEM_TYPE **dev_d_a, DATA_ITEM_TYPE **dev_d_c, 
			 DATA_ITEM_TYPE **dev_d_d, DATA_ITEM_TYPE **dev_d_e, 
			 DATA_ITEM_TYPE **dev_d_f, DATA_ITEM_TYPE **dev_d_h, 
			 DATA_ITEM_TYPE **dev_d_j, DATA_ITEM_TYPE **dev_d_k, 
			 DATA_ITEM_TYPE **dev_d_l, DATA_ITEM_TYPE **dev_d_m, 
			 DATA_ITEM_TYPE **dev_d_n, DATA_ITEM_TYPE **dev_d_o, 
			 DATA_ITEM_TYPE **dev_d_p, DATA_ITEM_TYPE **dev_d_q, 
			 int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1)
	objs_per_device = objs_per_device + trailing_objs;
      memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].r = dev_r[i];
      args[i].g = dev_g[i];
      args[i].b = dev_b[i];
      args[i].x = dev_x[i];
      args[i].a = dev_a[i];
      args[i].c = dev_c[i];
      args[i].d = dev_d[i];
      args[i].e = dev_e[i];
      args[i].f = dev_f[i];
      args[i].h = dev_h[i];
      args[i].j = dev_j[i];
      args[i].k = dev_k[i];
      args[i].l = dev_l[i];
      args[i].m = dev_m[i];
      args[i].d_r = dev_d_r[i];
      args[i].d_g = dev_d_g[i];
      args[i].d_b = dev_d_b[i];
      args[i].d_x = dev_d_x[i];
      args[i].d_a = dev_d_a[i];
      args[i].d_c = dev_d_c[i];
      args[i].d_d = dev_d_d[i];
      args[i].d_e = dev_d_e[i];
      args[i].d_f = dev_d_f[i];
      args[i].d_h = dev_d_h[i];
      args[i].d_j = dev_d_j[i];
      args[i].d_k = dev_d_k[i];
      args[i].d_l = dev_d_l[i];
      args[i].d_m = dev_d_m[i];
      args[i].d_n = dev_d_n[i];
      args[i].d_o = dev_d_o[i];
      args[i].d_p = dev_d_p[i];
      args[i].d_q = dev_d_q[i];
#else
      args[i].r = r[i];
      args[i].g = g[i];
      args[i].b = b[i];
      args[i].x = x[i];
      args[i].a = a[i];
      args[i].c = c[i];
      args[i].d = d[i];
      args[i].e = e[i];
      args[i].f = f[i];
      args[i].h = h[i];
      args[i].j = j[i];
      args[i].k = k[i];
      args[i].l = l[i];
      args[i].m = m[i];
      args[i].n = j[i];
      args[i].o = k[i];
      args[i].p = l[i];
      args[i].q = m[i];
      args[i].d_r = d_r[i];
      args[i].d_g = d_g[i];
      args[i].d_b = d_b[i];
      args[i].d_x = d_x[i];
      args[i].d_a = d_a[i];
      args[i].d_c = d_c[i];
      args[i].d_d = d_d[i];
      args[i].d_e = d_e[i];
      args[i].d_f = d_f[i];
      args[i].d_h = d_h[i];
      args[i].d_j = d_j[i];
      args[i].d_k = d_k[i];
      args[i].d_l = d_l[i];
      args[i].d_m = d_m[i];
      args[i].d_n = n[i];
      args[i].d_o = o[i];
      args[i].d_p = p[i];
      args[i].d_q = q[i];
#endif
      args[i].num_imgs = objs_per_device;
    }
    // reset this for rest of the program
    objs_per_device =  NUM_IMGS / gpu_agents_used;
}

void assign_brig_args_ca(brig_ca_arg *args, DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
			 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
			 int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].src_images = dev_src_images[i];
      args[i].dst_images = dev_dst_images[i];
#else
      args[i].src_images = src_images[i];
      args[i].dst_images = dst_images[i];
#endif
      args[i].num_imgs = objs_per_device;
    }
    // reset this for rest of the program
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}

void assign_brig_args_soa(brig_soa_arg *args, img **src_images, img **dst_images, 
			  img **dev_src_images, img **dev_dst_images, 
			  int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1)
        objs_per_device = objs_per_device + trailing_objs;
      memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].in = dev_src_images[i];
      args[i].out = dev_dst_images[i];
#else
      args[i].in = src_images[i];
      args[i].out = dst_images[i];
#endif
      args[i].num_imgs = objs_per_device;
    }
    objs_per_device =  NUM_IMGS / gpu_agents_used;

}

/********************************** 
 * 
 *    Argument assignment: AOS 
 *  
 **********************************/
void assign_gcn_args_aos(gcn_generic_arg *args, pixel **src_images, pixel **dst_images, 
			  pixel **dev_src_images, pixel **dev_dst_images, 
			  int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
    args[i].in = dev_src_images[i];
    args[i].out = dev_dst_images[i];
#else
    args[i].in = src_images[i];
    args[i].out = dev_dst_images[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  // reset this for rest of the program                                                                                
  objs_per_device =  NUM_IMGS / gpu_agents_used;
  return;
}

void assign_gcn_args_ca(gcn_generic_arg *args, DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images,
			DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images,
			int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;

  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
    args[i].in = dev_src_images[i];
    args[i].out = dev_dst_images[i];
#else
    args[i].in = src_images[i];
    args[i].out = dst_images[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}

void assign_gcn_args_copy_da(gcn_da_arg *args, 
			     DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **d_r,
			     DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_d_r,
			     int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].r = dev_r[i];
      args[i].d_r = dev_d_r[i];
#else
      args[i].r = r[i];
      args[i].d_r = d_r[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}

void assign_gcn_args_da_new(gcn_da_arg *args, 
			    DATA_ITEM_TYPE **r, 
			    DATA_ITEM_TYPE **d_r,
			    DATA_ITEM_TYPE **dev_r, 
			    DATA_ITEM_TYPE **dev_d_r, 
			    int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].r = dev_r[i];
      args[i].d_r = dev_d_r[i];
#else
      args[i].r = r[i];
      args[i].d_r = d_r[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}

#if 0
void assign_gcn_args_da(gcn_da_arg *args, 
			 DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
			 DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
			 DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
			 DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
			 DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
			 DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
			 DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
			 DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
			 DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
			 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
			 DATA_ITEM_TYPE **d_b, DATA_ITEM_TYPE **d_x, 
			 DATA_ITEM_TYPE **d_a, DATA_ITEM_TYPE **d_c, 
			 DATA_ITEM_TYPE **d_d, DATA_ITEM_TYPE **d_e, 
			 DATA_ITEM_TYPE **d_f, DATA_ITEM_TYPE **d_h, 
			 DATA_ITEM_TYPE **d_j, DATA_ITEM_TYPE **d_k, 
			 DATA_ITEM_TYPE **d_l, DATA_ITEM_TYPE **d_m, 
			 DATA_ITEM_TYPE **d_n, DATA_ITEM_TYPE **d_o, 
			 DATA_ITEM_TYPE **d_p, DATA_ITEM_TYPE **d_q, 
			 DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_g, 
			 DATA_ITEM_TYPE **dev_b, DATA_ITEM_TYPE **dev_x, 
			 DATA_ITEM_TYPE **dev_a, DATA_ITEM_TYPE **dev_c, 
			 DATA_ITEM_TYPE **dev_d, DATA_ITEM_TYPE **dev_e, 
			 DATA_ITEM_TYPE **dev_f, DATA_ITEM_TYPE **dev_h, 
			 DATA_ITEM_TYPE **dev_j, DATA_ITEM_TYPE **dev_k, 
			 DATA_ITEM_TYPE **dev_l, DATA_ITEM_TYPE **dev_m, 
			 DATA_ITEM_TYPE **dev_n, DATA_ITEM_TYPE **dev_o, 
			 DATA_ITEM_TYPE **dev_p, DATA_ITEM_TYPE **dev_q, 
			 DATA_ITEM_TYPE **dev_d_r, DATA_ITEM_TYPE **dev_d_g, 
			 DATA_ITEM_TYPE **dev_d_b, DATA_ITEM_TYPE **dev_d_x, 
			 DATA_ITEM_TYPE **dev_d_a, DATA_ITEM_TYPE **dev_d_c, 
			 DATA_ITEM_TYPE **dev_d_d, DATA_ITEM_TYPE **dev_d_e, 
			 DATA_ITEM_TYPE **dev_d_f, DATA_ITEM_TYPE **dev_d_h, 
			 DATA_ITEM_TYPE **dev_d_j, DATA_ITEM_TYPE **dev_d_k, 
			 DATA_ITEM_TYPE **dev_d_l, DATA_ITEM_TYPE **dev_d_m, 
			 DATA_ITEM_TYPE **dev_d_n, DATA_ITEM_TYPE **dev_d_o, 
			 DATA_ITEM_TYPE **dev_d_p, DATA_ITEM_TYPE **dev_d_q, 
			int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
#ifdef DEVMEM
      args[i].r = dev_r[i];
      args[i].g = dev_g[i];
      args[i].b = dev_b[i];
      args[i].x = dev_x[i];
      args[i].a = dev_a[i];
      args[i].c = dev_c[i];
      args[i].d = dev_d[i];
      args[i].e = dev_e[i];
      args[i].f = dev_f[i];
      args[i].h = dev_h[i];
      args[i].j = dev_j[i];
      args[i].k = dev_k[i];
      args[i].l = dev_l[i];
      args[i].m = dev_m[i];
      args[i].d_r = dev_d_r[i];
      args[i].d_g = dev_d_g[i];
      args[i].d_b = dev_d_b[i];
      args[i].d_x = dev_d_x[i];
      args[i].d_a = dev_d_a[i];
      args[i].d_c = dev_d_c[i];
      args[i].d_d = dev_d_d[i];
      args[i].d_e = dev_d_e[i];
      args[i].d_f = dev_d_f[i];
      args[i].d_h = dev_d_h[i];
      args[i].d_j = dev_d_j[i];
      args[i].d_k = dev_d_k[i];
      args[i].d_l = dev_d_l[i];
      args[i].d_m = dev_d_m[i];
      args[i].d_n = dev_d_n[i];
      args[i].d_o = dev_d_o[i];
      args[i].d_p = dev_d_p[i];
      args[i].d_q = dev_d_q[i];
#else
      args[i].r = r[i];
      args[i].g = g[i];
      args[i].b = b[i];
      args[i].x = x[i];
      args[i].a = a[i];
      args[i].c = c[i];
      args[i].d = d[i];
      args[i].e = e[i];
      args[i].f = f[i];
      args[i].h = h[i];
      args[i].j = j[i];
      args[i].k = k[i];
      args[i].l = l[i];
      args[i].m = m[i];
      args[i].n = j[i];
      args[i].o = k[i];
      args[i].p = l[i];
      args[i].q = m[i];
      args[i].d_r = d_r[i];
      args[i].d_g = d_g[i];
      args[i].d_b = d_b[i];
      args[i].d_x = d_x[i];
      args[i].d_a = d_a[i];
      args[i].d_c = d_c[i];
      args[i].d_d = d_d[i];
      args[i].d_e = d_e[i];
      args[i].d_f = d_f[i];
      args[i].d_h = d_h[i];
      args[i].d_j = d_j[i];
      args[i].d_k = d_k[i];
      args[i].d_l = d_l[i];
      args[i].d_m = d_m[i];
      args[i].d_n = n[i];
      args[i].d_o = o[i];
      args[i].d_p = p[i];
      args[i].d_q = q[i];
#endif
    args[i].num_imgs = objs_per_device;
  }
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}
#endif

void assign_gcn_args_soa(gcn_generic_arg *args, img **src_images, img **dst_images, 
			  int gpu_agents_used, int objs) {
  int i = 0;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;

  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1)
      objs_per_device = objs_per_device + trailing_objs;
    memset(&args[i], 0, sizeof(args[i]));
    args[i].in = src_images[i];
    args[i].out = dst_images[i];
    args[i].num_imgs = objs_per_device;
  }
  objs_per_device =  NUM_IMGS / gpu_agents_used;
}


/************************************* 
 *
 *      ALLOCATION 
 *
 *************************************/

void allocate_and_initialize_aos(pixel **src_images, pixel **dst_images, 
				 pixel **dev_src_images, pixel **dev_dst_images, 
				 hsa_agent_t *gpu_agents,
				 hsa_agent_t *cpu_agents, int gpu_agents_used,  int objs, int obj_size,
				 int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * obj_size;

    hsa_signal_value_t value;
    hsa_status_t err;
    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel); 
      }
      if (placement == PLACE_FINE || placement == PLACE_DEVMEM) {
	src_images[i] = (pixel *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	dst_images[i] = (pixel *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
      }
      if (placement == PLACE_COARSE) {
	src_images[i] = (pixel *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	dst_images[i] = (pixel *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
      }
      for (int j = 0; j < objs_per_device * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
	for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	  src_images[i][k].r = (DATA_ITEM_TYPE)k;
	  src_images[i][k].g = k * 10.0f;
#if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].b = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].x = k * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].a = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].c = k * 10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].d = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].e = k * 10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].f = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].h = k * 10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].j = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].k = k * 10.0f;
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].l = (DATA_ITEM_TYPE)k;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].m = k * 10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][k].n = k * 10.0f;
#endif
#if defined MEM16 || MEM17 || MEM18
	  src_images[i][k].o = k * 10.0f;
#endif
#if defined MEM17 || MEM18
	  src_images[i][k].p = k * 10.0f;
#endif
#if defined MEM18
	  src_images[i][k].q = k * 10.0f;
#endif
	}
    }
    // reset for next phase 
    objs_per_device =  objs / gpu_agents_used;
    segment_size = objs_per_device * obj_size; 

    return;
}


void allocate_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		 DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		 DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
		 DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
		 DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
		 DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
		 DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
		 DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
		 DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
		 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
		 DATA_ITEM_TYPE **d_b, DATA_ITEM_TYPE **d_x, 
		 DATA_ITEM_TYPE **d_a, DATA_ITEM_TYPE **d_c, 
		 DATA_ITEM_TYPE **d_d, DATA_ITEM_TYPE **d_e, 
		 DATA_ITEM_TYPE **d_f, DATA_ITEM_TYPE **d_h, 
		 DATA_ITEM_TYPE **d_j, DATA_ITEM_TYPE **d_k, 
		 DATA_ITEM_TYPE **d_l, DATA_ITEM_TYPE **d_m, 
		 DATA_ITEM_TYPE **d_n, DATA_ITEM_TYPE **d_o, 
		 DATA_ITEM_TYPE **d_p, DATA_ITEM_TYPE **d_q, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents,int gpu_agents_used, int objs, 
		 int obj_size, int placement) { 

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);

    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
      }
      if (placement == PLACE_FINE || placement == PLACE_DEVMEM) {
	r[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	g[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	b[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	x[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	a[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	c[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	e[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	f[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	h[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	j[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	k[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	l[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	m[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	n[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	o[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	p[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	q[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	
	d_r[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_g[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_b[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_x[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_a[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_c[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_d[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_e[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_f[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_h[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_j[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_k[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_l[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_m[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_n[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_o[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_p[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	d_q[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
      }
      if (placement == PLACE_COARSE) {
	r[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	g[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	b[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	x[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	a[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	c[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	e[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	f[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	h[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	j[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	k[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	l[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	m[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	n[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	o[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	p[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	q[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	
	d_r[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_g[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_b[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_x[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_a[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_c[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_d[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_e[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_f[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_h[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_j[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_k[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_l[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_m[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_n[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_o[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_p[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	d_q[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
      }

      if (!r[i] || !g[i] || !b[i] || !g[i] || !d_r[i] || !d_g[i] || !d_b[i] || !d_x[i]) {
	printf("Unable to malloc discrete arrays to fine grain memory. Exiting\n");
	exit(0);
      }
    }

}

void allocate_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;

    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
      }
      if (placement == PLACE_FINE || placement == PLACE_DEVMEM) {
	src_images[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	dst_images[i] = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
      }
      if (placement == PLACE_COARSE) {
	src_images[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	dst_images[i] = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
      }

      if (!src_images[i] || !dst_images[i]) {
	printf("Unable to malloc discrete arrays to fine grain memory. Exiting\n");
	exit(0);
      }

    }
    return ;
}

void allocate_soa(img **src_images, img **dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		  int placement) {

#if 0
    unsigned objs_per_device = objs / gpu_agents_used;
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * sizeof(img);

    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * sizeof(img);
    }
      if (placement == PLACE_FINE || placement == PLACE_DEVMEM) {
	src_images[i] = (img *) malloc_fine_grain_agent(gpu_agents[i], segment_size);      
	dst_images[i] = (img *) malloc_fine_grain_agent(gpu_agents[i], segment_size);
	
	for (int j = 0; j < objs_per_device; j++) {
	  src_images[i][j].r = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  src_images[i][j].g = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  src_images[i][j].b = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  src_images[i][j].x = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	
	  dst_images[i][j].r = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dst_images[i][j].g = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dst_images[i][j].b = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dst_images[i][j].x = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	}
      }
      if (placement == PLACE_COARSE) {
	src_images[i] = (img *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
	dst_images[i] = (img *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
      }
    }

#endif
    return;
}


/***************************
 *  
 *       COPY 
 *
 **************************/

void dev_copy_aos(pixel **src_images, pixel **dst_images, 
		  pixel **dev_src_images, pixel **dev_dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		  int placement) {   

  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel);
  
  hsa_status_t err;
  hsa_signal_value_t value;
  int i = 0;
  hsa_signal_t copy_sig[gpu_agents_used];
  
  if (placement == PLACE_DEVMEM) {
    hsa_signal_t copy_sig[gpu_agents_used];
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel); 
      }
      dev_src_images[i] = (pixel *) malloc_device_mem_agent(gpu_agents[i], segment_size);
      dev_dst_images[i] = (pixel *) malloc_device_mem_agent(gpu_agents[i], segment_size);
      if (!dev_src_images[i] || !dev_dst_images) {
	printf("Unable to malloc buffer to device memory. Exiting\n");
	exit(0);
      }
#ifdef VERBOSE
      printf("Successfully malloc'ed %lu MB to memory pool at %p and %p\n",
	     segment_size/(1024 * 1024), dev_src_images[i], dev_dst_images[i]);
#endif
      
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      
      // copy to device memory
      check(Creating a HSA signal, err);
      hsa_amd_memory_async_copy(dev_src_images[i], gpu_agents[i], src_images[i], gpu_agents[i],
            				segment_size, 0, NULL, copy_sig[i]);
      hsa_memory_copy(dev_src_images[i], src_images[i], segment_size);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
    }
  }
  
  return;
}


void dev_copy_aos_dst(pixel **src_images, pixel **dst_images, 
		  pixel **dev_src_images, pixel **dev_dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		      int placement, double *cp_time) {   

  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel);
  
  hsa_status_t err;
  hsa_signal_value_t value;
  int i = 0;
  hsa_signal_t copy_sig[gpu_agents_used];
  
  double cp_to_dev_time;

  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel); 
    }
    if (placement == PLACE_DEVMEM) {
      dev_src_images[i] = (pixel *) malloc_device_mem_agent(gpu_agents[i], segment_size);
    }
    dev_dst_images[i] = (pixel *) malloc_device_mem_agent(gpu_agents[i], segment_size);
    if (!dev_src_images[i] || !dev_dst_images) {
      printf("Unable to malloc buffer to device memory. Exiting\n");
      exit(0);
    }
#ifdef VERBOSE
    printf("Successfully malloc'ed %lu MB to memory pool at %p and %p\n",
	   segment_size/(1024 * 1024), dev_src_images[i], dev_dst_images[i]);
#endif
      
    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      
    // copy to device memory
    hsa_memory_copy(dev_dst_images[i], dst_images[i], segment_size);
    if (placement == PLACE_DEVMEM) {
      cp_to_dev_time = mysecond(); 
      check(Creating a HSA signal, err);
      hsa_amd_memory_async_copy(dev_src_images[i], gpu_agents[i], src_images[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      //      hsa_memory_copy(dev_src_images[i], src_images[i], segment_size);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
    				    HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
      cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time);
    }
  }

  (*cp_time) = cp_to_dev_time;
  return;
}
  
void host_copy_aos(pixel **src_images, pixel **dst_images, 
		  pixel **dev_src_images, pixel **dev_dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		  int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel);
    
    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    hsa_signal_t copy_sig[gpu_agents_used];
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(pixel); 
      }
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(dst_images[i], gpu_agents[i], dev_dst_images[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
    }
  return;
}

void host_copy_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    
    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    hsa_signal_t copy_sig[gpu_agents_used]; 
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
      }
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(dst_images[i], gpu_agents[i], dev_dst_images[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
    }
    return;
}

void dev_copy_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    
    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    if (placement == PLACE_DEVMEM) {
      hsa_signal_t copy_sig[gpu_agents_used];
      for (i = 0; i < gpu_agents_used; i++) {
	if (i == gpu_agents_used - 1) {
	  objs_per_device = objs_per_device + trailing_objs;
	  segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
	}
	
	dev_src_images[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
	dev_dst_images[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);	
	if (!dev_src_images[i] || !dev_dst_images) {
	  printf("Unable to malloc buffer to device memory. Exiting\n");
	  exit(0);
	}
#ifdef VERBOSE
	printf("Successfully malloc'ed %lu MB to memory pool at %p and %p\n",
	       segment_size/(1024 * 1024), dev_src_images[i], dev_dst_images[i]);
#endif

	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);

	/* copy to device memory */
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_src_images[i], gpu_agents[i], src_images[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_BLOCKED);
	err=hsa_signal_destroy(copy_sig[i]);
      }
    }
    return;
}

void dev_copy_soa(img **src_images, img **dst_images, 
		 img **dev_src_images, img **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {

#if 0
    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * sizeof(img);
    
    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    if (placement == PLACE_DEVMEM) {
      hsa_signal_t copy_sig[gpu_agents_used];
      for (i = 0; i < gpu_agents_used; i++) {
	if (i == gpu_agents_used - 1) {
	  objs_per_device = objs_per_device + trailing_objs;
	  segment_size = objs_per_device * sizeof(img);
	}
	dev_src_images[i] = (img *) malloc_device_mem_agent(gpu_agents[i], segment_size);
	dev_dst_images[i] = (img *) malloc_device_mem_agent(gpu_agents[i], segment_size);	
	if (!dev_src_images[i] || !dev_dst_images) {
	  printf("Unable to malloc buffer to device memory. Exiting\n");
	  exit(0);
	}
#ifdef VERBOSE
	printf("Successfully malloc'ed %lu MB to memory pool at %p and %p\n",
	       segment_size/(1024 * 1024), dev_src_images[i], dev_dst_images[i]);
#endif
	for (int j = 0; j < objs_per_device; j++) {
	  dev_src_images[i][j].r = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_src_images[i][j].g = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_src_images[i][j].b = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_src_images[i][j].x = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	
	  dev_dst_images[i][j].r = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_dst_images[i][j].g = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_dst_images[i][j].b = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	  dev_dst_images[i][j].x = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG);
	}

	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);

	/* copy to device memory */
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_src_images[i], gpu_agents[i], src_images[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_BLOCKED);
	err=hsa_signal_destroy(copy_sig[i]);
      }
    }
#endif
    return;
}

void dev_copy_da_allocate(DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_d_r, 
			  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
			  int gpu_agents_used, int objs, int obj_size, 
			  int placement) {

  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
  
  hsa_status_t err;
  hsa_signal_value_t value;
  int i = 0;
  hsa_signal_t copy_sig[gpu_agents_used];

  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
    }
    if (placement == PLACE_DEVMEM) {
      dev_r[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
      dev_d_r[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
    }
  }
  if (!dev_r[i] || !dev_d_r[i]) {
    printf("Unable to malloc discrete arrays to device memory. Exiting\n");
    exit(0);
  }
  return; 
}

void dev_copy_da_new(DATA_ITEM_TYPE **r,
		     DATA_ITEM_TYPE **dev_r, 
		     hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		     int gpu_agents_used, int objs, int obj_size, 
		     int placement) {
  
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
  
  hsa_status_t err;
  hsa_signal_value_t value;
  int i = 0;
  hsa_signal_t copy_sig[gpu_agents_used];

  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
    }

    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
    check(Creating a HSA signal, err);
    hsa_amd_memory_async_copy(dev_r[i], gpu_agents[i], r[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
    value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
				    HSA_WAIT_STATE_ACTIVE);
    err=hsa_signal_destroy(copy_sig[i]);

  }
}


void dev_copy_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		 DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		 DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
		 DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
		 DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
		 DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
		 DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
		 DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
		 DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
		 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
		 DATA_ITEM_TYPE **d_b, DATA_ITEM_TYPE **d_x, 
		 DATA_ITEM_TYPE **d_a, DATA_ITEM_TYPE **d_c, 
		 DATA_ITEM_TYPE **d_d, DATA_ITEM_TYPE **d_e, 
		 DATA_ITEM_TYPE **d_f, DATA_ITEM_TYPE **d_h, 
		 DATA_ITEM_TYPE **d_j, DATA_ITEM_TYPE **d_k, 
		 DATA_ITEM_TYPE **d_l, DATA_ITEM_TYPE **d_m, 
		 DATA_ITEM_TYPE **d_n, DATA_ITEM_TYPE **d_o, 
		 DATA_ITEM_TYPE **d_p, DATA_ITEM_TYPE **d_q, 
		 DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_g, 
		 DATA_ITEM_TYPE **dev_b, DATA_ITEM_TYPE **dev_x, 
		 DATA_ITEM_TYPE **dev_a, DATA_ITEM_TYPE **dev_c, 
		 DATA_ITEM_TYPE **dev_d, DATA_ITEM_TYPE **dev_e, 
		 DATA_ITEM_TYPE **dev_f, DATA_ITEM_TYPE **dev_h, 
		 DATA_ITEM_TYPE **dev_j, DATA_ITEM_TYPE **dev_k, 
		 DATA_ITEM_TYPE **dev_l, DATA_ITEM_TYPE **dev_m, 
		 DATA_ITEM_TYPE **dev_n, DATA_ITEM_TYPE **dev_o, 
		 DATA_ITEM_TYPE **dev_p, DATA_ITEM_TYPE **dev_q, 
		 DATA_ITEM_TYPE **dev_d_r, DATA_ITEM_TYPE **dev_d_g, 
		 DATA_ITEM_TYPE **dev_d_b, DATA_ITEM_TYPE **dev_d_x, 
		 DATA_ITEM_TYPE **dev_d_a, DATA_ITEM_TYPE **dev_d_c, 
		 DATA_ITEM_TYPE **dev_d_d, DATA_ITEM_TYPE **dev_d_e, 
		 DATA_ITEM_TYPE **dev_d_f, DATA_ITEM_TYPE **dev_d_h, 
		 DATA_ITEM_TYPE **dev_d_j, DATA_ITEM_TYPE **dev_d_k, 
		 DATA_ITEM_TYPE **dev_d_l, DATA_ITEM_TYPE **dev_d_m, 
		 DATA_ITEM_TYPE **dev_d_n, DATA_ITEM_TYPE **dev_d_o, 
		 DATA_ITEM_TYPE **dev_d_p, DATA_ITEM_TYPE **dev_d_q, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);

    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    hsa_signal_t copy_sig[gpu_agents_used];
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
      }
      if (placement == PLACE_DEVMEM) {
	dev_r[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
	dev_g[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
	dev_b[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
	
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_x[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_a[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_c[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_d[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_e[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_f[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_h[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_j[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_k[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_l[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	dev_m[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	dev_n[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM16 || MEM17 || MEM18
	dev_o[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM17 || MEM18
	dev_p[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM18
	dev_q[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
      }
      dev_d_r[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
      dev_d_g[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
      dev_d_b[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_x[i] =  (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_a[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_c[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_d[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_e[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_f[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
 #if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_h[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM11 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_j[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_k[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_l[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      dev_d_m[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      dev_d_n[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM16 || MEM17 || MEM18
      dev_d_o[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM17 || MEM18
      dev_d_p[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
#if defined MEM18
      dev_d_q[i] = (DATA_ITEM_TYPE *) malloc_device_mem_agent(gpu_agents[i], segment_size);
#endif
    
      if (placement == PLACE_DEVMEM) {
      
	if (!dev_r[i] || !dev_g[i] || !dev_b[i])  {
	  // || !dev_g[i] || !dev_d_r[i] 
	  //	   || !dev_d_g[i] || !dev_d_b[i] || !dev_d_x[i]) {
	  printf("Unable to malloc discrete arrays to device memory. Exiting\n");
	  exit(0);
	}
      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_r[i], gpu_agents[i], r[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
	
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_g[i], gpu_agents[i], g[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
	
	// #if defined MEM3 || MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_b[i], gpu_agents[i], b[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
	// #endif
	
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18     
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_x[i], gpu_agents[i], x[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_a[i], gpu_agents[i], a[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_c[i], gpu_agents[i], c[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_d[i], gpu_agents[i], d[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_e[i], gpu_agents[i], e[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_f[i], gpu_agents[i], f[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_h[i], gpu_agents[i], h[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_j[i], gpu_agents[i], j[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_k[i], gpu_agents[i], k[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_l[i], gpu_agents[i], l[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
	
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_m[i], gpu_agents[i], m[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_n[i], gpu_agents[i], n[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM16 || MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_o[i], gpu_agents[i], o[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM17 || MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_p[i], gpu_agents[i], p[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM18      
	err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
	check(Creating a HSA signal, err);
	hsa_amd_memory_async_copy(dev_q[i], gpu_agents[i], q[i], gpu_agents[i],
				  segment_size, 0, NULL, copy_sig[i]);
	value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
					HSA_WAIT_STATE_ACTIVE);
	err=hsa_signal_destroy(copy_sig[i]);
#endif
      }
    }
    return;
}

 
void host_copy_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		 DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		 DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
		 DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
		 DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
		 DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
		 DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
		 DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
		 DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
		 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
		 DATA_ITEM_TYPE **d_b, DATA_ITEM_TYPE **d_x, 
		 DATA_ITEM_TYPE **d_a, DATA_ITEM_TYPE **d_c, 
		 DATA_ITEM_TYPE **d_d, DATA_ITEM_TYPE **d_e, 
		 DATA_ITEM_TYPE **d_f, DATA_ITEM_TYPE **d_h, 
		 DATA_ITEM_TYPE **d_j, DATA_ITEM_TYPE **d_k, 
		 DATA_ITEM_TYPE **d_l, DATA_ITEM_TYPE **d_m, 
		 DATA_ITEM_TYPE **d_n, DATA_ITEM_TYPE **d_o, 
		 DATA_ITEM_TYPE **d_p, DATA_ITEM_TYPE **d_q, 
		 DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_g, 
		 DATA_ITEM_TYPE **dev_b, DATA_ITEM_TYPE **dev_x, 
		 DATA_ITEM_TYPE **dev_a, DATA_ITEM_TYPE **dev_c, 
		 DATA_ITEM_TYPE **dev_d, DATA_ITEM_TYPE **dev_e, 
		 DATA_ITEM_TYPE **dev_f, DATA_ITEM_TYPE **dev_h, 
		 DATA_ITEM_TYPE **dev_j, DATA_ITEM_TYPE **dev_k, 
		 DATA_ITEM_TYPE **dev_l, DATA_ITEM_TYPE **dev_m, 
		 DATA_ITEM_TYPE **dev_n, DATA_ITEM_TYPE **dev_o, 
		 DATA_ITEM_TYPE **dev_p, DATA_ITEM_TYPE **dev_q, 
		 DATA_ITEM_TYPE **dev_d_r, DATA_ITEM_TYPE **dev_d_g, 
		 DATA_ITEM_TYPE **dev_d_b, DATA_ITEM_TYPE **dev_d_x, 
		 DATA_ITEM_TYPE **dev_d_a, DATA_ITEM_TYPE **dev_d_c, 
		 DATA_ITEM_TYPE **dev_d_d, DATA_ITEM_TYPE **dev_d_e, 
		 DATA_ITEM_TYPE **dev_d_f, DATA_ITEM_TYPE **dev_d_h, 
		 DATA_ITEM_TYPE **dev_d_j, DATA_ITEM_TYPE **dev_d_k, 
		 DATA_ITEM_TYPE **dev_d_l, DATA_ITEM_TYPE **dev_d_m, 
		 DATA_ITEM_TYPE **dev_d_n, DATA_ITEM_TYPE **dev_d_o, 
		 DATA_ITEM_TYPE **dev_d_p, DATA_ITEM_TYPE **dev_d_q, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement) {
  
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
  
  hsa_status_t err;
  hsa_signal_value_t value;
  int i = 0;
  
  hsa_signal_t copy_sig[gpu_agents_used];
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
    }
    // Copy all device memory buffers back to host
    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
    check(Creating a HSA signal, err);
    
    hsa_amd_memory_async_copy(r[i], gpu_agents[i], dev_r[i], gpu_agents[i], segment_size, 0, NULL, copy_sig[i]);
    value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				    HSA_WAIT_STATE_BLOCKED);
    err=hsa_signal_destroy(copy_sig[i]);

    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
    check(Creating a HSA signal, err);

    hsa_amd_memory_async_copy(d_r[i], gpu_agents[i], dev_d_r[i], gpu_agents[i], segment_size, 0, NULL, copy_sig[i]);
    value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				    HSA_WAIT_STATE_BLOCKED);
    err=hsa_signal_destroy(copy_sig[i]);
    
    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
    check(Creating a HSA signal, err);
    
    hsa_amd_memory_async_copy(d_g[i], gpu_agents[i], dev_d_g[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
    value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				    HSA_WAIT_STATE_BLOCKED);
    err=hsa_signal_destroy(copy_sig[i]);
    
    err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
    check(Creating a HSA signal, err);      
    hsa_amd_memory_async_copy(d_b[i], gpu_agents[i], dev_d_b[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
    value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
                                    HSA_WAIT_STATE_BLOCKED);
    err=hsa_signal_destroy(copy_sig[i]);

    
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);      
      hsa_amd_memory_async_copy(d_x[i], gpu_agents[i], dev_d_x[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif

#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_a[i], gpu_agents[i], dev_d_a[i], gpu_agents[i], segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
      
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_c[i], gpu_agents[i], dev_d_c[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_d[i], gpu_agents[i], dev_d_d[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
                                    HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_e[i], gpu_agents[i], dev_d_e[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif

#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_f[i], gpu_agents[i], dev_d_f[i], gpu_agents[i], segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
      
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_h[i], gpu_agents[i], dev_d_h[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_j[i], gpu_agents[i], dev_d_j[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
                                    HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_k[i], gpu_agents[i], dev_d_k[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif

#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      hsa_amd_memory_async_copy(d_l[i], gpu_agents[i], dev_d_l[i], gpu_agents[i],
			      segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
                                    HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif

#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_m[i], gpu_agents[i], dev_d_m[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_n[i], gpu_agents[i], dev_d_n[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM16 || MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_o[i], gpu_agents[i], dev_d_o[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM17 || MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_p[i], gpu_agents[i], dev_d_p[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
#if defined MEM18
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(d_q[i], gpu_agents[i], dev_d_q[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
#endif
    }
  return;
}
/************************************* 
 *
 *      Initialization 
 *
 *************************************/
void initialize_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		   DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		   DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
		   DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
		   DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
		   DATA_ITEM_TYPE **j_data, DATA_ITEM_TYPE **k_data, 
		   DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
		   DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
		   DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
		   int gpu_agents_used, int objs, int obj_size, int placement) {

    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);

    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
      }
      for (int j = 0; j < objs_per_device * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
	for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	  r[i][k] = (DATA_ITEM_TYPE)k;
	  g[i][k] = k * 10.0f;
	  b[i][k] = (DATA_ITEM_TYPE)k;
	  x[i][k] = k * 10.0f;
	  a[i][k] = (DATA_ITEM_TYPE)k;
	  c[i][k] = k * 10.0f;
	  d[i][k] = (DATA_ITEM_TYPE)k;
	  e[i][k] = k * 10.0f;
	  f[i][k] = (DATA_ITEM_TYPE)k;
	  h[i][k] = k * 10.0f;
	  j_data[i][k] = (DATA_ITEM_TYPE)k;
	  k_data[i][k] = k * 10.0f;
	  l[i][k] = (DATA_ITEM_TYPE)k;
	  m[i][k] = k * 10.0f;
	  n[i][k] = (DATA_ITEM_TYPE)k;
	  o[i][k] = k * 10.0f;
	  p[i][k] = (DATA_ITEM_TYPE)k;
	  q[i][k] = k * 10.0f;
	}
    }
}

void initialize_ca(DATA_ITEM_TYPE **src_images, 
		   int gpu_agents_used, int objs, int obj_size, 
		   int placement) {

  int loc_sparsity = SPARSITY;
  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
  
  int i = 0;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    }
    int this_set_tile_count = 0;
    for (int j = 0, obj = 0; j < objs_per_device * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS), obj++)
      for (int k = j, t = 0; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS, t++) {
	for (int m = k, n = 0; m < k + TILE; m++, n++) { 	
	  if (t == loc_sparsity) {
	    this_set_tile_count++;
	    t = 0;
	  }
	  //	  int val = (n * loc_sparsity + t) + (this_set_tile_count * loc_sparsity * TILE);
	  int val = obj * PIXELS_PER_IMG + t * TILE + n;
	  src_images[i][OFFSET_R + m] = (DATA_ITEM_TYPE) val;
	  src_images[i][OFFSET_G + m] = val * 10.0f;
	  src_images[i][OFFSET_B + m] = (DATA_ITEM_TYPE)val;
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_X + m] = val * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_A + m] = (DATA_ITEM_TYPE) val;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_C + m] = val *  10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_D + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_E + m] = val *  10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_F + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_H + m] = val *  10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_J + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_K + m] = val *  10.0f;	  
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_L + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_M + m] = val *  10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_N + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM16 || MEM17 || MEM18
	  src_images[i][OFFSET_O + m] = val *  10.0f;	  
#endif
#if defined MEM17 || MEM18
	  src_images[i][OFFSET_P + m] = (DATA_ITEM_TYPE) val ;
#endif
#if defined MEM18
	  src_images[i][OFFSET_Q + m] = val *  10.0f;
#endif
}
      }
  }
}

void initialize_soa(img **src_images, 
		    int gpu_agents_used, int objs, int obj_size, 
		    int placement) {
  unsigned objs_per_device = objs / gpu_agents_used;
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * sizeof(img);
  
  int i = 0;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * sizeof(img);
    }
    for (int j = 0; j < objs_per_device; j++) 
      for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	src_images[i][j].r[k] = (DATA_ITEM_TYPE)k;
	src_images[i][j].g[k] = k * 10.0f;
#if 0 
	src_images[i][j].b[k] = (DATA_ITEM_TYPE)k;
	src_images[i][j].x[k] = k * 10.0f;
#endif
      }
  }
  return;
}


/********************************** 
 *
 * Data Layout Conversion Routines 
 *
 **********************************/
void convert_aos_to_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		       DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		       pixel **src_images, int gpu_agents_used, int objs, int obj_size) {
#if 0
    unsigned objs_per_device = objs / gpu_agents_used; 
    int trailing_objs = objs % gpu_agents_used;
    unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);

    hsa_status_t err;
    hsa_signal_value_t value;
    int i = 0;
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	objs_per_device = objs_per_device + trailing_objs;
	segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);
      }
      for (int j = 0; j < objs_per_device * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
	for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	  r[i][k] = src_images[i][k].r;
	  g[i][k] = src_images[i][k].g;
	  b[i][k] = src_images[i][k].b;
	  x[i][k] = src_images[i][k].x;
	}
    }
#endif
}

void convert_aos_to_ca(DATA_ITEM_TYPE **src_images_ca,
		       pixel **src_images, int gpu_agents_used, int objs, int obj_size) {

  unsigned objs_per_device = objs / gpu_agents_used; 
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
  int loc_sparsity = SPARSITY;
  
#if 0
  int i = 0;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    }
    for (int j = 0, m = 0; j < objs_per_device * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS), 
	   m += PIXELS_PER_IMG)
      for (int k = j, p = m; k < j + PIXELS_PER_IMG; k++, p++) {
	src_images_ca[i][OFFSET_R + k] = src_images[i][p].r;
	src_images_ca[i][OFFSET_G + k] = src_images[i][p].g;
	src_images_ca[i][OFFSET_B + k] = src_images[i][p].b;
	src_images_ca[i][OFFSET_X + k] = src_images[i][p].x;
      }
  }

#endif
  
  int i = 0;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    }

    int items_per_set = PIXELS_PER_IMG/SPARSITY;
    int index_reset_point = items_per_set/TILE;
    int this_set_tile_count = 0;
    for (int j = 0, obj = 0; j < objs_per_device * PIXELS_PER_IMG * FIELDS; 
	 j += (PIXELS_PER_IMG * FIELDS), obj++)
      for (int k = j, tilenum = 0; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS, tilenum++) {
	for (int m = k, n = 0; m < k + TILE; m++, n++) { 	
	  if (tilenum == index_reset_point) {
	    this_set_tile_count++;
	    tilenum = 0;
	  }
	  //	  int aos_loc = (n * loc_sparsity + tilenum) + (this_set_tile_count * loc_sparsity * TILE) + TILE * obj;
	  int aos_loc = (n + (tilenum * TILE) * loc_sparsity) + this_set_tile_count;
	  src_images_ca[i][OFFSET_R + m] = src_images[i][aos_loc].r;
	  src_images_ca[i][OFFSET_G + m] = src_images[i][aos_loc].g * 10.0f;
	  src_images_ca[i][OFFSET_B + m] = (DATA_ITEM_TYPE)src_images[i][aos_loc].b;
#if defined MEM4 || MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_X + m] = src_images[i][aos_loc].x * 10.0f;
#endif
#if defined MEM5 || MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_A + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].a;
#endif
#if defined MEM6 || MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_C + m] = src_images[i][aos_loc].c *  10.0f;
#endif
#if defined MEM7 || MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_D + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].d ;
#endif
#if defined MEM8 || MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_E + m] = src_images[i][aos_loc].e *  10.0f;
#endif
#if defined MEM9 || MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_F + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].f ;
#endif
#if defined MEM10 || MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_H + m] = src_images[i][aos_loc].h *  10.0f;
#endif
#if defined MEM11 || MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_J + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].h ;
#endif
#if defined MEM12 || MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_K + m] = src_images[i][aos_loc].k *  10.0f;	  
#endif
#if defined MEM13 || MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_L + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].l ;
#endif
#if defined MEM14 || MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_M + m] = src_images[i][aos_loc].m *  10.0f;
#endif
#if defined MEM15 || MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_N + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].n ;
#endif
#if defined MEM16 || MEM17 || MEM18
	  src_images_ca[i][OFFSET_O + m] = src_images[i][aos_loc].o *  10.0f;	  
#endif
#if defined MEM17 || MEM18
	  src_images_ca[i][OFFSET_P + m] = (DATA_ITEM_TYPE) src_images[i][aos_loc].p ;
#endif
#if defined MEM18
	  src_images_ca[i][OFFSET_Q + m] = src_images[i][aos_loc].q *  10.0f;
#endif
	}
      }
  }
}


void convert_aos_to_soa(img **src_images_soa, pixel **src_images,
			int gpu_agents_used, int objs, int obj_size) {
#if 0
  unsigned objs_per_device = objs / gpu_agents_used;
  int trailing_objs = objs % gpu_agents_used;
  unsigned long segment_size = objs_per_device * sizeof(img);
  
  int i = 0;
  for (i = 0; i < gpu_agents_used; i++) {
    if (i == gpu_agents_used - 1) {
      objs_per_device = objs_per_device + trailing_objs;
      segment_size = objs_per_device * sizeof(img);
    }
    for (int j = 0, m = 0; j < objs_per_device; j++, m += PIXELS_PER_IMG) 
      for (int k = j, p = m; k < j + PIXELS_PER_IMG; k++, p++) {
	src_images_soa[i][j].r[k] = src_images[i][p].r;
	src_images_soa[i][j].g[k] = src_images[i][p].g;
	src_images_soa[i][j].b[k] = src_images[i][p].b;
	src_images_soa[i][j].x[k] = src_images[i][p].x;
      }
  }
#endif
}



void convert_da_to_aos(DATA_ITEM_TYPE *d_r, DATA_ITEM_TYPE *d_g, DATA_ITEM_TYPE *d_b, 
		       DATA_ITEM_TYPE *d_x, pixel *dst_images, 
		       int start, int end) { 
#if 0
  for (int j = start * PIXELS_PER_IMG; j < end * PIXELS_PER_IMG; j += PIXELS_PER_IMG)     
   for (int i = j; i < j + PIXELS_PER_IMG; i++) {
     dst_images[i].r = d_r[i];
     dst_images[i].g = d_g[i];
     dst_images[i].b = d_b[i];
     dst_images[i].x = d_x[i];
   } 
#endif   
}





