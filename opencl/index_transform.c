#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

/* Problem size */
#define ITEMS 16
#define FIELDS 2
#define TILE 4
#define SPARSITY 2

#define ITEMS_PER_SET ITEMS/SPARSITY          
#define SET_COUNT_RESET ITEMS_PER_SET/TILE

//#define DEBUG  

typedef float DATA_TYPE;

typedef struct data_item_type {
  DATA_TYPE a;
  DATA_TYPE b;
} data_item;

void convert_aos_to_ca(data_item *aos_data, DATA_TYPE *ca) {
  int ref_set_count = 0;
  for (int j = 0, t = 0; j < ITEMS * FIELDS; j += TILE * FIELDS, t++)
    for (int i = 0, m = j; i < TILE; i++, m++) {
      if (t == SET_COUNT_RESET) {
	ref_set_count++;
	t = 0;
      }
      int aos_index = ((t * TILE + i) * SPARSITY) + ref_set_count; 

      printf("[%d]\t->\t[%d]\n", aos_index, m);

      ca[m] = aos_data[aos_index].a;
      ca[m + TILE] = aos_data[aos_index].b;
    }
}

void print_da(DATA_TYPE *A, DATA_TYPE *B) {
  for (int i = 0; i < ITEMS * ITEMS; i += ITEMS) {
    for (int j = i; j < i + ITEMS; j++) {
      printf("%3.2f\t", A[j]);
    }
    printf("\n");
  }
  return;
}

void check_ca_conversion(DATA_TYPE *A, DATA_TYPE *ca) {
  printf("checking CA conversion...\n");
  for (int i = 0, j = 0, t = 0; i < ITEMS * ITEMS; i++, j++, t++) {
    if (t == TILE) {
      j = j + TILE;
      t = 0;
    }
    if (A[i] != ca[j])
      printf("%f\t%f\n", A[i], ca[j]);
  }
  return;
}

void print_ca(DATA_TYPE *ca) {
  for (int i = 0; i < ITEMS * ITEMS * FIELDS; i += (TILE * FIELDS)) {
     for (int j = i; j < i + TILE; j++) {
       printf("[%d] %3.2f\t", i, ca[j]);
     }
     printf("\n");
  }
  return;
}

void gesummv_ca_kernel_debug(DATA_TYPE *ca, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
	
  for (int i = 0; i < ITEMS; i++) {
    int j;
    for(j = 0; j < ITEMS; j++) {	
      tmp[i] += ca[i + (j * ITEMS * FIELDS)] * x[j];
      y[i] += ca[i + (j * ITEMS * FIELDS) + TILE] * x[j];
      if (i < 4) 
	printf("%d\t", (i * FIELDS) - (i % TILE) * (FIELDS - 1) + (j * ITEMS * FIELDS));
    }
    if (i < 4) 
      printf("\n");
  }
  
  printf("\n");

  for (int i = 0; i < ITEMS; i++) {
    int j;
    for(j = 0; j < ITEMS; j++) {	
      tmp[i] += ca[i + (j * ITEMS * FIELDS)] * x[j];
      y[i] += ca[i + (j * ITEMS * FIELDS) + TILE] * x[j];
      if (i < 4) 
	printf("%d\t", (i * FIELDS) - (i % TILE) * (FIELDS - 1) + (j * ITEMS * FIELDS) + TILE);
    }
    if (i < 4) 
      printf("\n");
  }
  
  

}

#if 0
__global__ void gesummv_ca_kernel(DATA_TYPE *ca, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < ITEMS) {
    int j;
    int index = (i * FIELDS) - (i % TILE) * (FIELDS - 1);
    
    for(j = 0; j < ITEMS; j++) {	
      tmp[i] += ca[index + (j * ITEMS * FIELDS)] * x[j];
      y[i] += ca[index + (j * ITEMS * FIELDS) + TILE] * x[j];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}
#endif


void init(DATA_TYPE* A, DATA_TYPE* x)
{
  	int i, j;

 	for (i = 0; i < ITEMS; i++)
    {
    	x[i] = ((DATA_TYPE) i) / ITEMS;
      	
		for (j = 0; j < ITEMS; j++) 
		{
			A[i*ITEMS + j] = ((DATA_TYPE) i*j) / ITEMS;
		}
    }
}



int main(int argc, char *argv[]) {

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;
	

	DATA_TYPE *ca;
	data_item *aos;
	ca = (DATA_TYPE*)malloc(ITEMS*ITEMS*sizeof(DATA_TYPE) * FIELDS);
	aos = (data_item *) malloc(ITEMS*ITEMS*sizeof(data_item));

	A = (DATA_TYPE*)malloc(ITEMS*ITEMS*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(ITEMS*ITEMS*sizeof(DATA_TYPE));

	x = (DATA_TYPE*)malloc(ITEMS*sizeof(DATA_TYPE)); 
	init(A, x);

	for (int i = 0; i < ITEMS * ITEMS; i++) {
	  aos[i].a = A[i];
	  aos[i].b = B[i];
	}

	convert_aos_to_ca(aos, ca);

	//	check_ca_conversion(A, ca);
	//	print_ca(ca);
	//	print_da(A,B);

	return 0;
}

