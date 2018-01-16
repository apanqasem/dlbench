#include<stdio.h>

#define PIXELS_PER_IMG 64
#define TILE 4
#define N 1
#define SPARSITY 2
#define WORKGROUP 4

// N * T = PIXELS_PER_IMG
int main() {

  //  size_t i = get_local_id(0);
  //  size_t group_id = get_group_id(0);


  for (int j = 0; j < 4; j++) 
    for (int i = 0; i < TILE; i++) {
      int sets = (j / SPARSITY);    // sets processed 
      int set_offset = WORKGROUP * SPARSITY * sets;
      int index = (i * SPARSITY) + (j - SPARSITY * sets) + set_offset;
      printf("%d\t%d\t%d\t%d\t%d\n", j, i, sets, set_offset, index);
      
    }
  return 0;
}


#if 0    
  int r[PIXELS_PER_IMG];
  int d[PIXELS_PER_IMG];

  for (int i = 0; i < PIXELS_PER_IMG; i += TILE) {
    for (int j = i; j < i + TILE; j++) { 	
      r[j] = j;
    }
  }

  int refs_to_last_index = PIXELS_PER_IMG/N;
  int reset = refs_to_last_index / TILE;
  int cnt = 0;
  for (int i = 0, t = 0; i < PIXELS_PER_IMG; t++, i += TILE) {
    for (int j = i, n = 0; j < i + TILE; j++, n++) { 	
      if (t == N) {
	cnt++;
	t = 0;
      }
      d[j] = (n * N + t) + (cnt * N * TILE);
    }
  }

  for (int j = 0; j < PIXELS_PER_IMG; j++) {
    //    printf("r[%d] = %d\n", j, r[j]);
    printf("d[%d] = %d\n", j, d[j]);
  }
}

#endif
