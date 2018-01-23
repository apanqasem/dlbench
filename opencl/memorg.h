#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <argdefs.h>
#include <dlbench.h>

enum PLACEMENT { PLACE_FINE, PLACE_COARSE, PLACE_DEVMEM };

int check_agent_info(int *gpu_agents, int *cpu_agents); 
void get_agent_handles(hsa_agent_t *gpu_agents, hsa_agent_t *cpu_agents);

void allocate_and_initialize_aos(pixel **src_images, pixel **dst_images, 
				 pixel **dev_src_images, pixel **dev_dst_images, 
				 hsa_agent_t* gpu_agents, 
				 hsa_agent_t *cpu_agents, int gpu_agents_used, int objs, int obj_size, 
				 int placement);

void initialize_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		   DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		   DATA_ITEM_TYPE **a, DATA_ITEM_TYPE **c, 
		   DATA_ITEM_TYPE **d, DATA_ITEM_TYPE **e, 
		   DATA_ITEM_TYPE **f, DATA_ITEM_TYPE **h, 
		   DATA_ITEM_TYPE **j, DATA_ITEM_TYPE **k, 
		   DATA_ITEM_TYPE **l, DATA_ITEM_TYPE **m, 
		   DATA_ITEM_TYPE **n, DATA_ITEM_TYPE **o, 
		   DATA_ITEM_TYPE **p, DATA_ITEM_TYPE **q, 
		   int gpu_agents_used, int objs, int obj_size, int placement);

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
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents,int gpu_agents_used, int objs, int obj_size, int placement);

void allocate_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		  int placement);

void allocate_soa(img **src_images, img **dst_images, 
				hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
				int gpu_agents_used, int objs, int obj_size, 
				int placement);

void initialize_ca(DATA_ITEM_TYPE **src_images,
		  int gpu_agents_used, int objs, int obj_size, 
		  int placement);

void initialize_soa(img **src_images,
		    int gpu_agents_used, int objs, int obj_size, 
		    int placement);


void host_copy_aos(pixel **src_images, pixel **dst_images, 
		  pixel **dev_src_images, pixel **dev_dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		   int placement); 

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
		  int placement);

void host_copy_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		  int placement);


void dev_copy_aos(pixel **src_images, pixel **dst_images, 
		  pixel **dev_src_images, pixel **dev_dst_images, 
		  hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		  int gpu_agents_used, int objs, int obj_size, 
		   int placement); 

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
		 int placement);



void dev_copy_ca(DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
		 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		 int placement);

void dev_copy_soa(img **src_images, img **dst_images, 
		 img **dev_src_images, img **dev_dst_images, 
		 hsa_agent_t* gpu_agents, hsa_agent_t *cpu_agents, 
		 int gpu_agents_used, int objs, int obj_size, 
		  int placement);

void assign_brig_args_aos(brig_aos_arg *args, pixel **src_images, pixel **dst_images, 
			  pixel **dev_src_images, pixel **dev_dst_images, 
			  int gpu_agents_used, int objs); 

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
			 int gpu_agents_used, int objs);

void assign_brig_args_ca(brig_ca_arg *args, DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images, 
			 DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images, 
			 int gpu_agents_used, int objs);

void assign_brig_args_soa(brig_soa_arg *args, img **src_images, img **dst_images, 
			  img **dev_src_images, img **dev_dst_images, 
			  int gpu_agents_used, int objs);

void assign_gcn_args_aos(gcn_generic_arg *args, pixel **src_images, pixel **dst_images, 
			  pixel **dev_src_images, pixel **dev_dst_images, 
			  int gpu_agents_used, int objs); 

void assign_gcn_args_copy_da(gcn_da_arg *args, 
			     DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **d_r,
			     DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_d_r,
			     int gpu_agents_used, int objs);


void assign_gcn_args_da_new(gcn_da_arg *args, 
			 DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
			 DATA_ITEM_TYPE **b, 
			 DATA_ITEM_TYPE **d_r, DATA_ITEM_TYPE **d_g, 
			 DATA_ITEM_TYPE **d_b,
			 DATA_ITEM_TYPE **dev_r, DATA_ITEM_TYPE **dev_g, 
			 DATA_ITEM_TYPE **dev_b, 
			 DATA_ITEM_TYPE **dev_d_r, DATA_ITEM_TYPE **dev_d_g, 
			 DATA_ITEM_TYPE **dev_d_b, 
			    int gpu_agents_used, int objs);

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
			 int gpu_agents_used, int objs);

void assign_gcn_args_ca(gcn_generic_arg *args, DATA_ITEM_TYPE **src_images, DATA_ITEM_TYPE **dst_images,
			DATA_ITEM_TYPE **dev_src_images, DATA_ITEM_TYPE **dev_dst_images,
			int gpu_agents_used, int objs);


void assign_gcn_args_soa(gcn_generic_arg *args, img **src_images, img **dst_images, 
			  int gpu_agents_used, int objs);

/*
 * Layout conversion
 */
void convert_aos_to_da(DATA_ITEM_TYPE **r, DATA_ITEM_TYPE **g, 
		       DATA_ITEM_TYPE **b, DATA_ITEM_TYPE **x, 
		       pixel **src_images, int gpu_agents_used, int objs, int obj_size);

void convert_aos_to_ca(DATA_ITEM_TYPE **src_images_ca, pixel **src_images, 
		       int gpu_agents_used, int objs, int obj_size); 


void convert_aos_to_soa(img **src_images_soa, pixel **src_images,
			int gpu_agents_used, int objs, int obj_size);


void convert_da_to_aos(DATA_ITEM_TYPE *d_r, DATA_ITEM_TYPE *d_g, DATA_ITEM_TYPE *d_b, 
		       DATA_ITEM_TYPE *d_x, pixel *dst_images, 
		       int start, int end);

