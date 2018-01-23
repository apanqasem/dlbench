#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <pthread.h>
#include <mapi.h>
#include <memorg.h>
#include <util.h>
#include <host_kernels.h>
#include <argdefs.h>
#include <dlbench.h>

#include<da.h>

#define MAXNAME 256

#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed.\n", #msg); \
    exit(1); \
} 


int main(int argc, char *argv[]) {
    int gpu_agents_used;
    if (argc < 2)
      gpu_agents_used = 1;
    else 
      gpu_agents_used = atoi(argv[1]);

    int cpu_agents_used = 1;

    // this should come in as an argument 
    char kernel_base_name[MAXNAME]  = "dlbench";

    hsa_status_t err = hsa_init();
    check(Initializing the hsa runtime, err);

    int num_agents = check_agent_info(&gpu_agents_used, &cpu_agents_used);
    if (num_agents == 0) {               // did not find needed agents 
      err=hsa_shut_down();
      check(Shutting down the runtime, err);
      exit(1);  
    }
    /* Get a handle on all agents (whether we use them or not) */
    hsa_agent_t gpu_agents[num_agents];
    hsa_agent_t cpu_agents[num_agents];

    err = hsa_iterate_agents(get_all_gpu_agents, gpu_agents);
    check(Getting GPU agent handles, err);
    err = hsa_iterate_agents(get_all_cpu_agents, cpu_agents);
    check(Getting CPU agent handles, err);

    hsa_isa_t isas[gpu_agents_used];
    hsa_code_object_t code_objects[gpu_agents_used];
    hsa_executable_t executables[gpu_agents_used];

    char *module_file_name;
#ifdef BRIG 
    int module_type = 1;
    const char *ext = ".brig";
    module_file_name = strcat(kernel_base_name, ext);
#else
    int module_type = 0;
    const char *ext = ".hsaco";
    module_file_name = strcat(kernel_base_name, ext);
#endif    

    setup_hsa_kernel(gpu_agents_used, gpu_agents, module_type, module_file_name, 
		     isas, code_objects, executables);


    int i = 0;
    hsa_agent_t agent;
    hsa_executable_symbol_t symbols[gpu_agents_used];
   /*
    * Extract the symbol from the executable.
    */
    for (i = 0; i < gpu_agents_used; i++) {
#ifdef BRIG
#ifdef AOS
      err = hsa_executable_get_symbol(executables[i], NULL, "&__OpenCL_grayscale_aos_kernel",
      				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef DA
      err = hsa_executable_get_symbol(executables[i], NULL, "&__OpenCL_grayscale_da_kernel", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef SOA
      err = hsa_executable_get_symbol(executables[i], NULL, "&__OpenCL_grayscale_soa_kernel", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef CA
      err = hsa_executable_get_symbol(executables[i], NULL, "&__OpenCL_grayscale_ca_kernel", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#endif

#ifdef AMDGCN
#ifdef AOS
      err = hsa_executable_get_symbol(executables[i], NULL, "grayscale_aos", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef DA
      //      err = hsa_executable_get_symbol(executables[i], NULL, "grayscale_da_new", 
      //				      gpu_agents[i], 0, &symbols[i]);
      err = hsa_executable_get_symbol(executables[i], NULL, "copy_da", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef SOA
      err = hsa_executable_get_symbol(executables[i], NULL, "grayscale_soa", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#ifdef CA
      err = hsa_executable_get_symbol(executables[i], NULL, "grayscale_ca", 
				      gpu_agents[i], 0, &symbols[i]);
#endif
#endif
      check(Extract the symbol from the executable, err);
    } 
    /*
     * Extract dispatch information from the symbol
     */
    uint64_t kernel_objects[i];
    uint32_t kernarg_segment_sizes[i];
    uint32_t group_segment_sizes[i];
    uint32_t private_segment_sizes[i];

    for (i = 0; i < gpu_agents_used; i++) {
      err = hsa_executable_symbol_get_info(symbols[i], HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, 
					   &kernel_objects[i]);
      check(Extracting the symbol from the executable, err);
      err = hsa_executable_symbol_get_info(symbols[i], 
					   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, 
					   &kernarg_segment_sizes[i]);
      check(Extracting the kernarg segment size from the executable, err);
      err = hsa_executable_symbol_get_info(symbols[i], 
					   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, 
					   &group_segment_sizes[i]);
      check(Extracting the group segment size from the executable, err);
      err = hsa_executable_symbol_get_info(symbols[i], 
					   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, 
					   &private_segment_sizes[i]);
      check(Extracting the private segment from the executable, err);
    }

    /*
     * Create signals to wait for the dispatch to finish.
     */ 
    hsa_signal_t signals[gpu_agents_used];
    for (i = 0; i < gpu_agents_used; i++) {
      err=hsa_signal_create(1, 0, NULL, &signals[i]);
      check(Creating a HSA signal, err);
    }

    /* ********************************************
     * 
     * Data allocation and distribution code BEGIN 
     *
     *********************************************/

    int host_start = 0;
    int segments = 0;
    int num_cpu_agents = 1;

#ifdef FINE 
      int placement = PLACE_FINE;
#endif
#ifdef COARSE
      int placement = PLACE_COARSE;
#endif
#ifdef DEVMEM 
      int placement = PLACE_DEVMEM;
#endif

#ifdef HOST
    segments = num_cpu_agents;
#endif
#ifdef HETERO
    segments = num_cpu_agents + num_gpu_agents;
#endif

    hsa_signal_value_t value;

#ifdef HOST
    unsigned items_per_cpu_agent = NUM_IMGS / num_cpu_agents;
    unsigned trailing_items = NUM_IMGS % num_cpu_agents;
#if defined FINE || DEVMEM
#ifdef AOS
    /* allocate buffers in fine-grain memory region, accessible by all agents */
    pixel *src_images_host[segments];
    pixel *dst_images_host[segments];
    unsigned long segment_size = items_per_cpu_agent * PIXELS_PER_IMG * sizeof(pixel);

    for (int i = 0; i < segments; i++) {
      if (i == segments - 1) {
	items_per_cpu_agent = items_per_cpu_agent + trailing_items;
	segment_size = items_per_cpu_agent * PIXELS_PER_IMG * sizeof(pixel);
      }
#if defined FINE || DEVMEM
      src_images_host[i] = (pixel *) malloc_fine_grain_agent(cpu_agents[0], segment_size);
      dst_images_host[i] = (pixel *) malloc_fine_grain_agent(cpu_agents[0], segment_size);
      //            src_images_host[i] = (pixel *) malloc(segment_size);
      //      dst_images_host[i] = (pixel *) malloc(segment_size);
#endif
#ifdef COARSE
      src_images[i] = (pixel *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
      dst_images[i] = (pixel *) malloc_coarse_grain_agent(cpu_agents[0], segment_size);
#endif
      for (int j = 0; j < items_per_cpu_agent * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
	for (int k = j; k < j + PIXELS_PER_IMG; k++) {
	  src_images_host[i][k].r = (DATA_ITEM_TYPE)k;
	  src_images_host[i][k].g = k * 10.0f;
	  src_images_host[i][k].b = (DATA_ITEM_TYPE)k;
#if (MEM >= 4) 
	  src_images_host[i][k].x = k * 10.0f;
#endif
#if (MEM >= 5)
	  src_images_host[i][k].a = (DATA_ITEM_TYPE)k;
#endif
#if (MEM >= 6)
	  src_images_host[i][k].c = k * 10.0f;
#endif
#if (MEM >= 7)
	  src_images_host[i][k].d = (DATA_ITEM_TYPE)k;
#endif
#if (MEM >= 8)
	  src_images_host[i][k].e = k * 10.0f;
#endif
#if (MEM >= 9)
	  src_images_host[i][k].f = (DATA_ITEM_TYPE)k;
#endif
#if (MEM >= 10)
	  src_images_host[i][k].h = k * 10.0f;
#endif
	}
    }
    // reset for next phase                                                                                 
    items_per_cpu_agent =  NUM_IMGS / num_cpu_agents;
    segment_size = items_per_cpu_agent * PIXELS_PER_IMG * sizeof(pixel);
#endif
#ifdef DA
    unsigned long size_host = sizeof(DATA_ITEM_TYPE) * PIXELS_PER_IMG * NUM_IMGS;

#if defined FINE || DEVMEM
    DATA_ITEM_TYPE *r_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *g_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *b_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *x_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);

    DATA_ITEM_TYPE *d_r_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_g_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_b_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_x_host = (DATA_ITEM_TYPE *) malloc_fine_grain_agent(cpu_agents[0], size_host);
#endif
#ifdef COARSE
    DATA_ITEM_TYPE *r_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *g_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *b_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *x_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);

    DATA_ITEM_TYPE *d_r_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_g_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_b_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
    DATA_ITEM_TYPE *d_x_host = (DATA_ITEM_TYPE *) malloc_coarse_grain_agent(cpu_agents[0], size_host);
#endif

    if (!r_host || !g_host || !b_host || !g_host || !d_r_host || !d_g_host || !d_b_host || !d_x_host) {
      printf("Unable to malloc discrete arrays to fine/coarse grain memory. Exiting\n");
      exit(0);
    }

    for (int j = 0; j < items_per_cpu_agent * PIXELS_PER_IMG; j += PIXELS_PER_IMG) {
       for (int k = j; k < j + PIXELS_PER_IMG; k++) {
         r_host[k] = (DATA_ITEM_TYPE)k;
	 g_host[k] = k * 10.0f;
	 b_host[k] = (DATA_ITEM_TYPE)k;
	 x_host[k] = k * 10.0f;
       }
    }
#endif

      double t_host = mysecond();
      pthread_t threads[CPU_THREADS];

      unsigned int num_imgs_per_thread = ((NUM_IMGS - host_start)/CPU_THREADS);

      // TODO: when parallelizing by pixel, need different implementation for HETERO                      
   
      unsigned pixels_per_thread = (PIXELS_PER_IMG / CPU_THREADS);
      int start = host_start;
      int device_end = NUM_IMGS;
#ifdef AOS
      args_aos host_args[CPU_THREADS];
#else
      args_da host_args[CPU_THREADS];
#endif
      for (int i = 0; i < CPU_THREADS; i++) {
        host_args[i].start_index = start;
	host_args[i].end_index = start + pixels_per_thread;
	//      host_args[i].end_index = start + num_imgs_per_thread;
#ifdef AOS
      // entire array passed to each thread; should consider passing sections                               
      host_args[i].src = src_images_host[0];
      host_args[i].dst = dst_images_host[0];
      //      pthread_create(&threads[i], NULL, &host_grayscale_aos, (void *) &host_args[i]);
      pthread_create(&threads[i], NULL, &host_grayscale_aos_pixel, (void *) &host_args[i]);
#else
      host_args[i].r = r_host;
      host_args[i].g = g_host;
      host_args[i].b = b_host;
      host_args[i].x = x_host;
      host_args[i].d_r = d_r_host;
      host_args[i].d_g = d_g_host;
      host_args[i].d_b = d_b_host;
      host_args[i].d_x = d_x_host;
      pthread_create(&threads[i], NULL, &host_grayscale_da_pixel, (void *) &host_args[i]);
#endif
      start = start + pixels_per_thread;
      //      start = start + num_imgs_per_thread; 
    }
      for (int i = 0; i < CPU_THREADS; i++) {
      pthread_join(threads[i], NULL);
    }
      t_host = 1.0E6 * (mysecond() - t_host);

#endif
#endif

#ifdef DEVICE
    int obj_size;
    int objs = NUM_IMGS;
    double cp_to_dev_time, cp_to_host_time;
    double conv_time; 

#if defined AOS || COPY
    pixel *src_images[gpu_agents_used]; 
    pixel *dst_images[gpu_agents_used]; 
    obj_size = PIXELS_PER_IMG * sizeof(pixel);

    pixel *dev_src_images[gpu_agents_used];
    pixel *dev_dst_images[gpu_agents_used];

    allocate_and_initialize_aos(src_images, dst_images, dev_src_images, dev_dst_images, gpu_agents, 
				cpu_agents, gpu_agents_used, objs, obj_size, placement);

    //    print_aos(src_images[0], NUM_IMGS); 
    dev_copy_aos_dst(src_images, dst_images, dev_src_images, dev_dst_images, 
		     gpu_agents, cpu_agents, 
		     gpu_agents_used, objs, obj_size, placement, &cp_to_dev_time);
    
/* #ifdef DEVMEM */
/*     cp_to_dev_time = mysecond(); */
/*     dev_copy_aos(src_images, dst_images, dev_src_images, dev_dst_images,  */
/* 		  gpu_agents, cpu_agents,  */
/* 		  gpu_agents_used, objs, obj_size, placement); */
/*     cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time); */
/* #endif */

#endif  // END AOS

#ifdef DA
    DATA_ITEM_TYPE *r[gpu_agents_used];
    DATA_ITEM_TYPE *g[gpu_agents_used];
    DATA_ITEM_TYPE *b[gpu_agents_used];
    DATA_ITEM_TYPE *x[gpu_agents_used];
    DATA_ITEM_TYPE *a[gpu_agents_used];
    DATA_ITEM_TYPE *c[gpu_agents_used];
    DATA_ITEM_TYPE *d[gpu_agents_used];
    DATA_ITEM_TYPE *e[gpu_agents_used];
    DATA_ITEM_TYPE *f[gpu_agents_used];
    DATA_ITEM_TYPE *h[gpu_agents_used];
    DATA_ITEM_TYPE *j[gpu_agents_used];
    DATA_ITEM_TYPE *k[gpu_agents_used];
    DATA_ITEM_TYPE *l[gpu_agents_used];
    DATA_ITEM_TYPE *m[gpu_agents_used];
    DATA_ITEM_TYPE *n[gpu_agents_used];
    DATA_ITEM_TYPE *o[gpu_agents_used];
    DATA_ITEM_TYPE *p[gpu_agents_used];
    DATA_ITEM_TYPE *q[gpu_agents_used];

    DATA_ITEM_TYPE *d_r[gpu_agents_used];
    DATA_ITEM_TYPE *d_g[gpu_agents_used];
    DATA_ITEM_TYPE *d_b[gpu_agents_used];
    DATA_ITEM_TYPE *d_x[gpu_agents_used];
    DATA_ITEM_TYPE *d_a[gpu_agents_used];
    DATA_ITEM_TYPE *d_c[gpu_agents_used];
    DATA_ITEM_TYPE *d_d[gpu_agents_used];
    DATA_ITEM_TYPE *d_e[gpu_agents_used];
    DATA_ITEM_TYPE *d_f[gpu_agents_used];
    DATA_ITEM_TYPE *d_h[gpu_agents_used];
    DATA_ITEM_TYPE *d_j[gpu_agents_used];
    DATA_ITEM_TYPE *d_k[gpu_agents_used];
    DATA_ITEM_TYPE *d_l[gpu_agents_used];
    DATA_ITEM_TYPE *d_m[gpu_agents_used];
    DATA_ITEM_TYPE *d_n[gpu_agents_used];
    DATA_ITEM_TYPE *d_o[gpu_agents_used];
    DATA_ITEM_TYPE *d_p[gpu_agents_used];
    DATA_ITEM_TYPE *d_q[gpu_agents_used];

    DATA_ITEM_TYPE *dev_r[gpu_agents_used];
    DATA_ITEM_TYPE *dev_g[gpu_agents_used];
    DATA_ITEM_TYPE *dev_b[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_x[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_a[gpu_agents_used];
    DATA_ITEM_TYPE *dev_c[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d[gpu_agents_used];
    DATA_ITEM_TYPE *dev_e[gpu_agents_used];
    DATA_ITEM_TYPE *dev_f[gpu_agents_used];
    DATA_ITEM_TYPE *dev_h[gpu_agents_used];
    DATA_ITEM_TYPE *dev_j[gpu_agents_used];
    DATA_ITEM_TYPE *dev_k[gpu_agents_used];
    DATA_ITEM_TYPE *dev_l[gpu_agents_used];
    DATA_ITEM_TYPE *dev_m[gpu_agents_used];
    DATA_ITEM_TYPE *dev_n[gpu_agents_used];
    DATA_ITEM_TYPE *dev_o[gpu_agents_used];
    DATA_ITEM_TYPE *dev_p[gpu_agents_used];
    DATA_ITEM_TYPE *dev_q[gpu_agents_used];

    DATA_ITEM_TYPE *dev_d_r[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_d_g[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_d_b[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_d_x[gpu_agents_used]; 
    DATA_ITEM_TYPE *dev_d_a[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_c[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_d[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_e[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_f[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_h[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_j[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_k[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_l[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_m[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_n[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_o[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_p[gpu_agents_used];
    DATA_ITEM_TYPE *dev_d_q[gpu_agents_used];

    obj_size = PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE);

    allocate_da(r, g, b, x, 
		a, c, d, e, f, h, j, k, l, m, n, o, p, q,
		d_r, d_g, d_b, d_x, 
		d_a, d_c, d_d, d_e, d_f, d_h, d_j, d_k, d_l, d_m, d_n, d_o, d_p, d_q,
		gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);

#ifdef COPY 
    cp_to_dev_time = mysecond();
    convert_aos_to_da(r, g, b, x, src_images, gpu_agents_used, objs, obj_size);
    cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time);
#else
    initialize_da(r, g, b, x, 
		  a, c, d, e, f, h, j, k, l, m, n, o, p, q,
		gpu_agents_used, objs, obj_size, placement);


#endif
#ifdef DEVMEM
    cp_to_dev_time = mysecond();
    dev_copy_da(r, g, b, x, a, c, d, e, f, h, j, k, l, m, n, o, p, q,
		d_r, d_g, d_b, d_x, d_a, d_c, d_d, d_e, d_f, d_h, d_j, d_k, d_l, d_m, d_n, d_o, d_p, d_q, 
		dev_r, dev_g, dev_b, dev_x,
		dev_a, dev_c, dev_d, dev_e, 
		dev_f, dev_h, dev_j, dev_k, 
		dev_l, dev_m, dev_n, dev_o, dev_p, dev_q,
		dev_d_r, dev_d_g, dev_d_b, dev_d_x, 
		dev_d_a, dev_d_c, dev_d_d, dev_d_e, 
		dev_d_f, dev_d_h, dev_d_j, dev_d_k, 
		dev_d_l, dev_d_m, dev_d_n, dev_d_o, dev_d_p, dev_d_q,
		gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);
    cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time);
#endif
#endif // END DA 
#ifdef CA
    DATA_ITEM_TYPE *src_images_ca[gpu_agents_used];
    DATA_ITEM_TYPE *dst_images_ca[gpu_agents_used];
    DATA_ITEM_TYPE *dev_src_images_ca[gpu_agents_used];
    DATA_ITEM_TYPE *dev_dst_images_ca[gpu_agents_used];

    obj_size = PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS;
    allocate_ca(src_images_ca, dst_images_ca, gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);
#ifdef COPY
    conv_time = mysecond();
    convert_aos_to_ca(src_images_ca, src_images, gpu_agents_used, objs, obj_size);
    conv_time = 1.0E6 * (mysecond() - conv_time);
#else 
    initialize_ca(src_images_ca, gpu_agents_used, objs, obj_size, placement);
#endif
    //    print_ca(src_images_ca[0], host_start, objs);
#ifdef DEVMEM
    cp_to_dev_time = mysecond();
    dev_copy_ca(src_images_ca, dst_images_ca, dev_src_images_ca, dev_dst_images_ca, 
		gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);
    cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time);

#endif 
#endif // END CA  
#ifdef SOA
    /* img **src_images_soa = (img **) malloc_fine_grain_agent(cpu_agents[0], sizeof (img *) * gpu_agents_used); */
    /* img **dst_images_soa = (img **) malloc_fine_grain_agent(cpu_agents[0], sizeof (img *) * gpu_agents_used); */
    img *src_images_soa[gpu_agents_used];
    img *dst_images_soa[gpu_agents_used];
    img *dev_src_images_soa[gpu_agents_used];
    img *dev_dst_images_soa[gpu_agents_used];
    obj_size = sizeof(img);
    allocate_soa(src_images_soa, dst_images_soa, gpu_agents, cpu_agents, 
		 gpu_agents_used, objs, obj_size, placement);
#ifdef COPY 
    cp_to_dev_time = mysecond();
    convert_aos_to_soa(src_images_soa, src_images, gpu_agents_used, objs, obj_size);
    cp_to_dev_time = 1.0E6 * (mysecond() - cp_to_dev_time);
#else 
    initialize_soa(src_images_soa, gpu_agents_used, objs, obj_size, placement);
#endif
#ifdef DEVMEM
    dev_copy_soa(src_images_soa, dst_images_soa, dev_src_images_soa, dev_dst_images_soa, 
		gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);
#endif
#endif // END SOA
 

#ifdef BRIG
#ifdef AOS
    brig_aos_arg args[gpu_agents_used];
    assign_brig_args_aos(args, src_images, dst_images, dev_src_images, dev_dst_images, 
			 gpu_agents_used, objs);
#endif  
#ifdef DA
    brig_da_arg args[gpu_agents_used];
    assign_brig_args_da(args, 
			r, g, b, x, a, c, d, e, f, h, j, k, l, m, n, o, p, q,
			d_r, d_g, d_b, d_x, d_a, d_c, d_d, d_e, d_f, d_h, d_j, d_k, d_l, d_m, d_n, d_o, d_p, d_q, 
			dev_r,dev_g,dev_b,dev_x, dev_a, dev_c, dev_d, dev_e, dev_f, dev_h, 
			dev_j, dev_k, dev_l, dev_m, dev_n, dev_o, dev_p, dev_q, 
			dev_d_r,dev_d_g,dev_d_b,dev_d_x,dev_d_a, dev_d_c, dev_d_d, dev_d_e, dev_d_f, dev_d_h, 
			dev_d_j, dev_d_k, dev_d_l, dev_d_m, dev_d_n, dev_d_o, dev_d_p, dev_d_q, 
			gpu_agents_used, objs);
#endif
#ifdef CA
    brig_ca_arg args[gpu_agents_used];
    assign_brig_args_ca(args, src_images_ca, dst_images_ca, dev_src_images_ca, dev_dst_images_ca, 
			gpu_agents_used, objs); 
#endif
#ifdef SOA
    brig_soa_arg args[gpu_agents_used];
    assign_brig_args_soa(args, src_images_soa, dst_images_soa, dev_src_images_soa, dev_dst_images_soa, 
			gpu_agents_used, objs); 
#endif
#endif 


#ifdef AMDGCN
#ifdef AOS
    gcn_generic_arg args[gpu_agents_used];
    assign_gcn_args_aos(args, src_images, dst_images, dev_src_images, dev_dst_images, 
			gpu_agents_used, objs);
#endif  
#ifdef DA
    gcn_da_arg args[gpu_agents_used];
#if 0
        assign_gcn_args_da(args, 
			r, g, b, x, a, c, d, e, f, h, j, k, l, m, n, o, p, q,
			d_r, d_g, d_b, d_x, d_a, d_c, d_d, d_e, d_f, d_h, d_j, d_k, d_l, d_m, d_n, d_o, d_p, d_q, 
			dev_r,dev_g,dev_b,dev_x, dev_a, dev_c, dev_d, dev_e, dev_f, dev_h, 
			dev_j, dev_k, dev_l, dev_m, dev_n, dev_o, dev_p, dev_q, 
			dev_d_r,dev_d_g,dev_d_b,dev_d_x,dev_d_a, dev_d_c, dev_d_d, dev_d_e, dev_d_f, dev_d_h, 
			dev_d_j, dev_d_k, dev_d_l, dev_d_m, dev_d_n, dev_d_o, dev_d_p, dev_d_q, 
			gpu_agents_used, objs);
#endif
    /* assign_gcn_args_da_new(args,  */
    /* 			   r, g, b, */
    /* 			   d_r, d_g, d_b,  */
    /* 			   dev_r,dev_g,dev_b, */
    /* 			   dev_d_r,dev_d_g,dev_d_b, */
    /* 			   gpu_agents_used, objs); */
    assign_gcn_args_copy_da(args, 
			    r, d_r, dev_r, dev_d_r,
			    gpu_agents_used, objs);
#endif
#ifdef CA
    gcn_generic_arg args[gpu_agents_used];
    assign_gcn_args_ca(args, src_images_ca, dst_images_ca, dev_src_images_ca, dev_dst_images_ca, 
			gpu_agents_used, objs); 
#endif
#ifdef SOA
    gcn_generic_arg args[gpu_agents_used];
    assign_gcn_args_soa(args, src_images_soa, dst_images_soa, gpu_agents_used, objs);
#endif
#endif // END AMDGCN

    /********************************************
    
    Data allocation and distribution code END

   *********************************************/

    double t;
    //#ifdef C2GI

    void* kernarg_addresses[gpu_agents_used]; 
    uint32_t queue_sizes[gpu_agents_used];
    hsa_queue_t* queues[gpu_agents_used]; 
    for (int k = 0; k < KERNEL_ITERS; k++) {    
      // Allocate (and copy) kernel arguments 
      for (i = 0; i < gpu_agents_used; i++) {
	kernarg_addresses[i] = malloc_kernarg_agent(gpu_agents[i],kernarg_segment_sizes[i]);
	memcpy(kernarg_addresses[i], &args[i], sizeof(args[i]));
      }
      
      // Query maximum size queue for each GPU agent.    
      for (i = 0; i < gpu_agents_used; i++) {
	err = hsa_agent_get_info(gpu_agents[i], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_sizes[i]);
	check(Querying the agent maximum queue size, err);
      }
      
      // Create queues using the maximum size.
      for (i = 0; i < gpu_agents_used; i++) {
	err = hsa_queue_create(gpu_agents[i], queue_sizes[i], HSA_QUEUE_TYPE_SINGLE, 
			       NULL, NULL, UINT32_MAX, UINT32_MAX, &queues[i]);
	check(Creating queues, err);
      }
      // Obtain queue write indices.
      uint64_t indices[gpu_agents_used];
      for (i = 0; i < gpu_agents_used; i++) {
	indices[i] = hsa_queue_load_write_index_relaxed(queues[i]);
      }
      
      // Write the aql packet at the calculated queue index address.
      hsa_kernel_dispatch_packet_t* dispatch_packets[gpu_agents_used];
      for (i = 0; i < gpu_agents_used; i++) {
	const uint32_t queueMask = queues[i]->size - 1;
	dispatch_packets[i] = &(((hsa_kernel_dispatch_packet_t*) 
				 (queues[i]->base_address))[indices[i]&queueMask]);
	
	dispatch_packets[i]->setup  |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
	dispatch_packets[i]->workgroup_size_x = (uint16_t) WORKGROUP;
	dispatch_packets[i]->workgroup_size_y = (uint16_t)1;
	dispatch_packets[i]->workgroup_size_z = (uint16_t)1;
	dispatch_packets[i]->grid_size_x = (uint32_t) (THREADS);
	dispatch_packets[i]->grid_size_y = 1;
	dispatch_packets[i]->grid_size_z = 1;
	dispatch_packets[i]->completion_signal = signals[i];
	dispatch_packets[i]->kernel_object = kernel_objects[i];
	dispatch_packets[i]->kernarg_address = (void*) kernarg_addresses[i];
	dispatch_packets[i]->private_segment_size = private_segment_sizes[i];
	dispatch_packets[i]->group_segment_size = group_segment_sizes[i];
	
	uint16_t header = 0;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
	header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
	
	__atomic_store_n((uint16_t*)(&dispatch_packets[i]->header), header, __ATOMIC_RELEASE);
      }
      
      // Increment the write index and ring the doorbell to dispatch the kernel.
      for (i = 0; i < gpu_agents_used; i++) {
	hsa_queue_store_write_index_relaxed(queues[i], indices[i]+1);
	hsa_signal_store_relaxed(queues[i]->doorbell_signal, indices[i]);
	check(Dispatching the kernel, err);
      }
      t = mysecond();
      // Wait on the dispatch completion signal until the kernel is finished.
      for (i = 0; i < gpu_agents_used; i++) {
	value = hsa_signal_wait_acquire(signals[i], HSA_SIGNAL_CONDITION_LT, 1, 
					UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
      }
      t = 1.0E6 * (mysecond() - t);
    }
/* #endif */
/* #ifdef C2G */
/*       // Increment the write index and ring the doorbell to dispatch the kernel. */
/*       for (i = 0; i < gpu_agents_used; i++) { */
/* 	hsa_queue_store_write_index_relaxed(queues[i], indices[i]+1); */
/* 	hsa_signal_store_relaxed(queues[i]->doorbell_signal, indices[i]); */
/* 	check(Dispatching the kernel, err); */
/*       } */
/*       // Wait on the dispatch completion signal until the kernel is finished. */
/*       for (i = 0; i < gpu_agents_used; i++) { */
/* 	value = hsa_signal_wait_acquire(signals[i], HSA_SIGNAL_CONDITION_LT, 1,  */
/* 					UINT64_MAX, HSA_WAIT_STATE_ACTIVE); */
/*       } */
/*     t = 1.0E6 * (mysecond() - t); */
/* #endif */

    unsigned items_per_device = NUM_IMGS / gpu_agents_used; 
    int trailing_items = NUM_IMGS % gpu_agents_used;
    unsigned long segment_size;

#ifdef DEVMEM
    hsa_signal_t copy_sig[gpu_agents_used];
#ifdef AOS
    cp_to_host_time = mysecond();
    host_copy_aos(src_images, dst_images, dev_src_images, dev_dst_images, 
		  gpu_agents, cpu_agents, 
		  gpu_agents_used, objs, obj_size, placement);
    cp_to_host_time = 1.0E6 * (mysecond() - cp_to_host_time);
#endif
#ifdef DA
    cp_to_host_time = mysecond();
    host_copy_da(r, g, b, x, a, c, d, e, f, h, j, k, l, m, n, o, p, q,
		 d_r, d_g, d_b, d_x, d_a, d_c, d_d, d_e, d_f, d_h, d_j, d_k, d_l, d_m, d_n, d_o, d_p, d_q, 
		 dev_r, dev_g, dev_b, dev_x,
		 dev_a, dev_c, dev_d, dev_e, 
		 dev_f, dev_h, dev_j, dev_k, 
		 dev_l, dev_m, dev_n, dev_o, dev_p, dev_q,
		 dev_d_r, dev_d_g, dev_d_b, dev_d_x, 
		 dev_d_a, dev_d_c, dev_d_d, dev_d_e, 
		 dev_d_f, dev_d_h, dev_d_j, dev_d_k, 
		 dev_d_l, dev_d_m, dev_d_n, dev_d_o, dev_d_p, dev_d_q,
		 gpu_agents, cpu_agents, 
		 gpu_agents_used, objs, obj_size, placement);
    cp_to_host_time = 1.0E6 * (mysecond() - cp_to_host_time);
#endif // END DA 
#ifdef CA
    cp_to_host_time = mysecond();
    host_copy_ca(src_images_ca, dst_images_ca, dev_src_images_ca, dev_dst_images_ca, 
		gpu_agents, cpu_agents, 
		gpu_agents_used, objs, obj_size, placement);
    cp_to_host_time = 1.0E6 * (mysecond() - cp_to_host_time);
#endif
#ifdef SOA
    segment_size = items_per_device * sizeof(img);
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1) {
	items_per_device = items_per_device + trailing_items;
	segment_size = items_per_device * PIXELS_PER_IMG * sizeof(pixel); 
      }
      err=hsa_signal_create(1, 0, NULL, &copy_sig[i]);
      check(Creating a HSA signal, err);
      
      hsa_amd_memory_async_copy(dst_images_soa[i], gpu_agents[i], dev_dst_images_soa[i], gpu_agents[i],
				segment_size, 0, NULL, copy_sig[i]);
      value = hsa_signal_wait_acquire(copy_sig[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, 
				      HSA_WAIT_STATE_BLOCKED);
      err=hsa_signal_destroy(copy_sig[i]);
    }
    // reset for next phase 
    items_per_device =  NUM_IMGS / gpu_agents_used;
    segment_size = items_per_device * PIXELS_PER_IMG * sizeof(pixel);
#endif
#endif // END DEVMEM
#endif // END DEVICE 

#ifdef HOST
#ifdef DA
    check_results_da(r_host, g_host, b_host, x_host, d_r_host, host_start, items_per_cpu_agent);
#endif
    for (i = 0; i < segments; i++) {
      if (i == segments - 1)
	items_per_cpu_agent = items_per_cpu_agent + trailing_items;
#ifdef AOS
      check_results_aos(src_images_host[i], dst_images_host[i], host_start, items_per_cpu_agent);
#endif
    }
#ifdef CA
    check_results_ca(src_images_host, dst_images_host, host_start, device_end);
#endif
#endif

#ifdef DEVICE 
#ifdef AOS
    /* for (i = 0; i < gpu_agents_used; i++) { */
    /*   if (i == gpu_agents_used - 1) */
    /* 	items_per_device = items_per_device + trailing_items; */
    /*   check_results_aos(src_images[i], dst_images[i], host_start, items_per_device); */
    /* } */
#endif
#ifdef SOA
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1)
    	items_per_device = items_per_device + trailing_items;
      check_results_soa(src_images_soa[i], dst_images_soa[i], host_start, items_per_device);
    }
#endif
#ifdef DA
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1)
	items_per_device = items_per_device + trailing_items;
      check_results_da(r[i], g[i], b[i], x[i], d_r[i], host_start, items_per_device);
    }
#endif
#ifdef CA
    for (i = 0; i < gpu_agents_used; i++) {
      if (i == gpu_agents_used - 1)
	items_per_device = items_per_device + trailing_items;
      check_results_ca(src_images_ca[i], dst_images_ca[i], host_start, items_per_device);
    }
#endif
#endif
#ifdef HOST
    double secs = t_host/1000000;
#else 
    double secs = (double) t/1000000;
    double secs_copy = cp_to_host_time/1000000;
#endif

    /*   
     * Calculate performance metrics                                                                   
     */
    unsigned long input_data = NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS; 
    unsigned long output_data = NUM_IMGS * PIXELS_PER_IMG * sizeof(DATA_ITEM_TYPE) * FIELDS; 
    unsigned long data_transfer = input_data + output_data; 
    float dataGB = (float) data_transfer / 1e+09;

    // FIJI
    double FLOP = ((NUM_IMGS * 30) * PIXELS_PER_IMG) + (16 * PIXELS_PER_IMG);
    // adjust for unroll factor 
    FLOP = FLOP + ((float) (ITERS - 1) * (float) NUM_IMGS * (float) PIXELS_PER_IMG);
    double gFLOP = FLOP / 1e+09;
    
    float throughput = gFLOP /secs;
    float throughput_with_copy = gFLOP /(secs + (cp_to_dev_time/1000000));
    // unsigned FLOP = 28 * PIXELS_PER_IMG * NUM_IMGS;

#ifdef VERBOSE
    fprintf(stdout, "Kernel execution time %3.2f ms\n", t/1000);
    fprintf(stdout, "Copy to device time: %3.2f ms\n", cp_to_dev_time/1000);
    fprintf(stdout, "Copy to host time: %3.2f ms\n", cp_to_host_time/1000);
    fprintf(stdout, "Bandwidth: %3.2f GB/s\n", dataGB/ (float) secs);
    fprintf(stdout, "FLOP/s: %3.2f GB/s\n", throughput);
    fprintf(stdout, "Arithmetic intensity: %3.2f\n", FLOP/((float)data_transfer));
#else
#ifdef HOST
    fprintf(stdout, "%3.2f\n", t_host/1000);
#else 
    fprintf(stdout, "%3.2f", t/1000); 
#if defined COARSE || FINE
   fprintf(stdout, ",%3.2f\n", throughput);
#else
    fprintf(stdout, ",%3.2f,%3.2f,%3.2f\n",
	    cp_to_dev_time/1000,
	    throughput, 
	    throughput_with_copy);

#endif

#endif
#endif

#ifdef HOST 
    for (i = 0; i < segments; i++) {
#ifdef AOS
#ifdef FINE
      free_fine_grain(src_images_host[i]);
      free_fine_grain(dst_images_host[i]);
      //      free(src_images_host[i]);
      //      free(dst_images_host[i]);
#endif
#endif
#ifdef DA
#ifdef FINE
    free_fine_grain(r_host);
    free_fine_grain(g_host);
    free_fine_grain(b_host);
    free_fine_grain(x_host);
    free_fine_grain(d_r_host);
    free_fine_grain(d_g_host);
    free_fine_grain(d_b_host);
    free_fine_grain(d_x_host);
#endif
#endif
    }
#endif


#ifdef DEVICE
    /*
     * Cleanup all allocated resources.
     */
    for (i = 0; i < gpu_agents_used; i++) {
      err = hsa_memory_free(kernarg_addresses[i]);
      check(Freeing kernel argument memory buffer, err);
      
      err=hsa_signal_destroy(signals[i]);
      check(Destroying the signal, err);
      
      err=hsa_queue_destroy(queues[i]);
      check(Destroying the queue, err);

#ifdef AOS
#ifdef FINE
      free_fine_grain(src_images[i]);
      free_fine_grain(dst_images[i]);
#endif
#ifdef COARSE
      free_coarse_grain(src_images[i]);
      free_coarse_grain(dst_images[i]);
#endif
#ifdef DEVMEM
      free_device_mem(dev_src_images[i]);
      free_device_mem(dev_dst_images[i]);
#endif
#endif

#ifdef SOA
#ifdef FINE
      free_fine_grain(src_images_soa[i]);
      free_fine_grain(dst_images_soa[i]);
#endif
#ifdef COARSE
      free_coarse_grain(src_images_soa[i]);
      free_coarse_grain(dst_images_soa[i]);
#endif
#ifdef DEVMEM
      free_device_mem(dev_src_images_soa[i]);
      free_device_mem(dev_dst_images_soa[i]);
#endif
#endif

#ifdef CA
#ifdef FINE
      free_fine_grain(src_images_ca[i]);
      free_fine_grain(dst_images_ca[i]);
#endif
#ifdef COARSE
      free_coarse_grain(src_images_ca[i]);
      free_coarse_grain(dst_images_ca[i]);
#endif
#ifdef DEVMEM
      free_device_mem(dev_src_images_ca[i]);
      free_device_mem(dev_dst_images_ca[i]);
#endif
#endif

#ifdef DA
#ifdef FINE
      free_fine_grain(r[i]);
      free_fine_grain(g[i]);
      free_fine_grain(b[i]);
#if 0
      free_fine_grain(x[i]);
      free_fine_grain(d_r[i]);
      free_fine_grain(d_g[i]);
      free_fine_grain(d_b[i]);
      free_fine_grain(d_x[i]);
#endif
#endif
#ifdef COARSE
      free_coarse_grain(r[i]);
      free_coarse_grain(g[i]);
      free_coarse_grain(b[i]);
      free_coarse_grain(x[i]);
      free_coarse_grain(d_r[i]);
      free_coarse_grain(d_g[i]);
      free_coarse_grain(d_b[i]);
      free_coarse_grain(d_x[i]);
#endif
#ifdef DEVMEM
      free_device_mem(dev_r[i]);
      free_device_mem(dev_g[i]);
      free_device_mem(dev_b[i]);
#if 0
      free_device_mem(dev_x[i]);
      free_device_mem(dev_d_r[i]);
      free_device_mem(dev_d_g[i]);
      free_device_mem(dev_d_b[i]);
      free_device_mem(dev_d_x[i]);
#endif
#endif
#endif
    err=hsa_executable_destroy(executables[i]);
    check(Destroying the executable, err);
    
    err=hsa_code_object_destroy(code_objects[i]);
    check(Destroying the code object, err);
    }
#endif 

#ifndef SOA
    err=hsa_shut_down();
    check(Shutting down the runtime, err);
#endif
    return 0;
}
