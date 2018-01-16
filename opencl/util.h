#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <dlbench.h>
double mysecond();

int load_module_from_file(const char* file_name, hsa_ext_module_t* module, size_t* size);

void setup_hsa_kernel(int gpu_agents_used, hsa_agent_t *gpu_agents, int module_type, 
		      char* module_filename, 
		      hsa_isa_t *isas, hsa_code_object_t *code_objects, hsa_executable_t *executables);


void check_results_aos(pixel *src_images, pixel *dst_images, int host_start, int device_end);

void check_results_soa(img *src_images, img *dst_images, int host_start, int device_end);

void check_results_da(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,
                      DATA_ITEM_TYPE *d_r, int host_start, int device_end);

void check_results_ca(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images, 
                      int host_start, int device_end);

void print_ca(DATA_ITEM_TYPE *src_images, int host_start, int device_end);
void print_aos(pixel *src_images, int device_end); 
