#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <util.h>
#include <hsa.h>
#include <hsa_ext_finalize.h>

#define MAX_ERRORS 10
#define check(msg, status) \
if (status != HSA_STATUS_SUCCESS) { \
    printf("%s failed.\n", #msg); \
    exit(1); \
} 

double mysecond() {
  struct timeval tp;
  int i;

  i = gettimeofday(&tp,NULL);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

/*                                                                                                
 * Loads a BRIG module from a specified file. This                                               
 * function does not validate the module.                                                        
 */
int load_module_from_file(const char* file_name, hsa_ext_module_t* module, size_t* size) {
  int rc = -1;
  FILE *fp = fopen(file_name, "rb");
  if (!fp) {
    printf("Could not fine module file. Exiting\n");
    exit(0);
  }

  rc = fseek(fp, 0, SEEK_END);
  size_t file_size = (size_t) (ftell(fp) * sizeof(char));
  rc = fseek(fp, 0, SEEK_SET);
  char* buf = (char*) malloc(file_size);
  memset(buf,0,file_size);
  size_t read_size = fread(buf,sizeof(char),file_size,fp);

  if(read_size != file_size) {
    free(buf);
  } else {
    rc = 0;
    *module = (hsa_ext_module_t) buf;
    (*size) = file_size;
  }

  fclose(fp);

  return rc;
}
void check_results_aos(pixel *src_images, pixel *dst_images, int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
#ifdef HOST
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
	      src_images[i].b) / (F1 * src_images[i].b);
	v1 = v0 - F1 * src_images[i].b;
      }
      DATA_ITEM_TYPE exp_result = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 

#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[512].r);
#endif
      float delta = fabs(dst_images[i].r - exp_result);
      
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#else
  for (int j = 0; j < device_end * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[i].r / src_images[i].g + (F0 + F1 * F1) * 
		   src_images[i].b) / (F1 * src_images[i].b);
	v1 = v0 - F1 * src_images[i].b;
      }
      DATA_ITEM_TYPE exp_result = (src_images[i].r * v0 - src_images[i].g * F1 * v1); 

#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[i].r);
#endif
      float delta = fabs(dst_images[i].r - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%d %f %f\n", i, exp_result, dst_images[i].r);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
#endif
}

void check_results_da(DATA_ITEM_TYPE *r, DATA_ITEM_TYPE *g, DATA_ITEM_TYPE *b, DATA_ITEM_TYPE *x,
                      DATA_ITEM_TYPE *d_r, int host_start, int device_end) {

  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
#ifdef HOST
  for (int j =  host_start * PIXELS_PER_IMG; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG) {
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
	v1 = v0 - F1 * b[i];
      }
      DATA_ITEM_TYPE exp_result = (r[i] * v0 - g[i] * F1 * v1);
#ifdef DEBUG
	if (i == 512)
	  printf("%3.2f %3.2f\n", exp_result, d_r[i]);
#endif
      float delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
	  errors++;
#ifdef VERBOSE
	  if (errors < MAX_ERRORS) 
	    printf("%d %f %f\n", i, exp_result, d_r[i]);
#endif
	}
    }
  }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (CPU)" : "PASSED (CPU)"));
#endif

#ifdef DEVICE 
  for (int j = 0; j < device_end * PIXELS_PER_IMG; j += PIXELS_PER_IMG)
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (r[i] / g[i] + (F0 + F1 * F1) * b[i]) / (F1 * b[i]);
	v1 = v0 - F1 * b[i];
      }
      DATA_ITEM_TYPE exp_result = (r[i] * v0 - g[i] * F1 * v1);

#ifdef DEBUG
      if (i == 512)    // check a pixel in the middle of the image
        printf("%f %f\n", exp_result, d_r[i]);
#endif
      float delta = fabs(d_r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%d %f %f\n", i, exp_result, d_r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
  #endif 
  return;
}


void print_aos(pixel *src_images, int device_end) {
  for (int j =  0, img = 0; j < NUM_IMGS * PIXELS_PER_IMG; j += PIXELS_PER_IMG, img++)
    for (unsigned int i = j, p = 0; i < j + PIXELS_PER_IMG; i++, p++) {
      printf("%d\t%d\t%3.2f\n", img, p, src_images[i].r);
    }
}

void print_ca(DATA_ITEM_TYPE *src_images, int host_start, int device_end) {
  for (int j = 0, img = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS), img++)
    for (int k = j, t = 0; k < j + (PIXELS_PER_IMG * FIELDS); t++, k += TILE * FIELDS)
      for (int m = k, p = t * TILE; m < k + TILE; m++, p++) 
	printf("%d\t%d\t%d\t%3.2f\t%d\n", img, t, p, src_images[OFFSET_R + m], m);
   
}

void check_results_ca(DATA_ITEM_TYPE *src_images, DATA_ITEM_TYPE *dst_images, 
                      int host_start, int device_end) {
  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
      for (int k = j; k < j + (PIXELS_PER_IMG * FIELDS); k += TILE * FIELDS) {
	for (int m = k; m < k + TILE; m++) { 
	  DATA_ITEM_TYPE v0 = 0.0f;
	  DATA_ITEM_TYPE v1 = 0.0f;
	  for (int k = 0; k < ITERS; k++) {
	    v0 = v0 + (src_images[OFFSET_R + m] / src_images[OFFSET_G + m] + (F0 + F1 * F1) 
		       * src_images[OFFSET_B + m]) / (F1 * src_images[OFFSET_B + m]);
	    v1 = v0 - F1 * src_images[OFFSET_B + m];
	  }
	  DATA_ITEM_TYPE exp_result = 
	    (src_images[OFFSET_R + m] * v0 - src_images[OFFSET_G + m] * F1 * v1);
#ifdef DEBUG
	  if (m == 512)
	    printf("%f %f\n", exp_result, dst_images[OFFSET_R + m]);
	  //      printf("%f %f %f\n", dst_images[OFFSET_R + m], dst_images[OFFSET_G + m], dst_images[OFFSET_B + m]);
#endif
	  float delta = fabs(dst_images[OFFSET_R + m] - exp_result);
	  if (delta/exp_result > ERROR_THRESH) {
	    errors++;
#ifdef DEBUG
	    if (errors < MAX_ERRORS)
	      printf("%f %f\n", exp_result, dst_images[m]);
#endif
	  }
	}
      }

#if 0
  for (int j = 0; j < device_end * PIXELS_PER_IMG * FIELDS; j += (PIXELS_PER_IMG * FIELDS))
    for (unsigned int i = j; i < j + PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[OFFSET_R + i] / src_images[OFFSET_G + i] + (F0 + F1 * F1) 
		   * src_images[OFFSET_B + i]) / (F1 * src_images[OFFSET_B + i]);
	v1 = v0 - F1 * src_images[OFFSET_B + i];
      }
      DATA_ITEM_TYPE exp_result = (src_images[OFFSET_R + i] * v0 - src_images[OFFSET_G + i] * F1 * v1);
#ifdef DEBUG
      if (i == 512)
      printf("%f %f\n", exp_result, dst_images[OFFSET_B + i]);
      printf("%f %f %f\n", dst_images[OFFSET_R + i], dst_images[OFFSET_G + i], dst_images[OFFSET_B + i]);
#endif
      float delta = fabs(dst_images[OFFSET_R + i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
	  printf("%f %f\n", exp_result, dst_images[i]);
#endif
      }
    }
#endif
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


void check_results_soa(img *src_images, img *dst_images, int host_start, int device_end) {

#if 0
  float F0 = 0.02f; 
  float F1 = 0.30f; 
  int errors = 0;
  for (int j = 0; j < device_end; j++) 
    for (unsigned int i = 0; i < PIXELS_PER_IMG; i++) {
      DATA_ITEM_TYPE v0 = 0.0f;
      DATA_ITEM_TYPE v1 = 0.0f;
      for (int k = 0; k < ITERS; k++) {
	v0 = v0 + (src_images[j].r[i] / src_images[j].g[i] + (F0 + F1 * F1) * 
		   src_images[j].b[i]) / (F1 * src_images[j].b[i]);
	v1 = v0 - F1 * src_images[j].b[i];
      }
      DATA_ITEM_TYPE exp_result = (src_images[j].r[i] * v0 - src_images[j].g[i] * F1 * v1);
#ifdef DEBUG
      if (i == 512)
	printf("%3.2f %3.2f\n", exp_result, dst_images[j].r[i]);
#endif
      float delta = fabs(dst_images[j].r[i] - exp_result);
      if (delta/exp_result > ERROR_THRESH) {
        errors++;
#ifdef DEBUG
        if (errors < MAX_ERRORS)
          printf("%d %f %f\n", i, exp_result, dst_images[j].r[i]);
#endif
      }
    }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
#endif
}


 void setup_hsa_kernel(int gpu_agents_used, hsa_agent_t *gpu_agents, int module_type, char *filename,
		      hsa_isa_t *isas, hsa_code_object_t *code_objects, hsa_executable_t *executables) {
    /*
     * Determine if the finalizer 1.0 extension is supported.
     */
    hsa_status_t err;
    int i = 0;

    bool support;
    err = hsa_system_extension_supported(HSA_EXTENSION_FINALIZER, 1, 0, &support);
    check(Checking finalizer 1.0 extension support, err);

    /*
     * Generate the finalizer function table.
     */
    hsa_ext_finalizer_1_00_pfn_t table_1_00;
    err = hsa_system_get_extension_table(HSA_EXTENSION_FINALIZER, 1, 0, &table_1_00);
    check(Generating function table for finalizer, err);

    /*
     * Obtain GPU machine model
     */
    hsa_machine_model_t machine_models[gpu_agents_used];
    hsa_profile_t profiles[gpu_agents_used];
    for (i = 0; i < gpu_agents_used; i++) {
      err = hsa_agent_get_info(gpu_agents[i], HSA_AGENT_INFO_MACHINE_MODEL, &machine_models[i]);
      check("Obtaining machine model",err);
      err = hsa_agent_get_info(gpu_agents[i], HSA_AGENT_INFO_PROFILE, &profiles[i]);
      check("Getting agent profile",err);
      err = hsa_agent_get_info(gpu_agents[i], HSA_AGENT_INFO_ISA, &isas[i]);
      check(Query the agents isa, err);
    }

    hsa_ext_module_t module; 
    size_t size;

    if (module_type) {
      /*
       * Get module from file 
       */
      load_module_from_file(filename, &module, &size);
      /*
       * Create hsa program.
       */
      hsa_ext_program_t program;
      memset(&program, 0, sizeof(hsa_ext_program_t));
      err = table_1_00.hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, 
					      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, 
					      NULL, &program);
      check(Create the program, err);


      hsa_ext_control_directives_t control_directives[gpu_agents_used];
      /* 
       * Add module to program 
       */
      err = table_1_00.hsa_ext_program_add_module(program, module);
      /*
       * Finalize the program and extract the code object.
       */
      
      /* must create multiple code objects, otherwise cannot execute on multiple agents */
      for (i = 0; i < gpu_agents_used; i++) {
	memset(&control_directives[i], 0, sizeof(hsa_ext_control_directives_t));
	err = table_1_00.hsa_ext_program_finalize(program, isas[i], 0, control_directives[i], 
						  "", HSA_CODE_OBJECT_TYPE_PROGRAM, &code_objects[i]);
	check(Finalizing the program, err);
      }
      
      /*
       * Destroy the program, it is no longer needed.
       */
      err=table_1_00.hsa_ext_program_destroy(program);
    }
    else {
      /*
       * Get module from file 
       */
      load_module_from_file(filename, &module, &size);
      for (i = 0; i < gpu_agents_used; i++) {
	err = hsa_code_object_deserialize(module, size, NULL, &code_objects[i]);
	check(Deserializing amdgcn module, err);
      }
    }
      
    /*
     * Create the empty executable.
     */
    for (i = 0; i < gpu_agents_used; i++) {
      err = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &executables[i]);
      check(Create the executable, err);
      err = hsa_executable_load_code_object(executables[i], gpu_agents[i], code_objects[i], "");
      check(Loading the code object, err);
      err = hsa_executable_freeze(executables[i], "");
      check(Freeze the executable, err);
    }
    return;
}

