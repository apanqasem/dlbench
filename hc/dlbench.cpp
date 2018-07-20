#include <dlbench.h>
#include <kernel.h>

#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include <codecvt>
#include <locale>
#include <numeric>

#include <hc_am.hpp>

#define VEGA 2  // vega card is device 2 on paripp2

// global timers
// need these as global so as not to increase the number of parameters
// in the function invoking the GPU kernel

double t_cp_in_a = 0;
double t_cp_in_b = 0;
double t_cp_in_c = 0;
double t_cp_in_d = 0;
double t_cp_in_e = 0;
double t_cp_in_f = 0;
double t_cp_in_g = 0;
double t_cp_in_h = 0;
double t_cp_out = 0;
double t_kernel = 0;
double t_par_loop = 0;


std::string getDeviceName(const hc::accelerator& _acc) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string value = converter.to_bytes(_acc.get_description());
  return value;
}

void listDevices(void) {
  // Get number of devices
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();

  // Print device names
  if (accs.empty())
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < accs.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(accs[i]) << std::endl;
    }
    std::cout << std::endl;
  }
}

void check_results(DATA_ITEM_TYPE *in_a, DATA_ITEM_TYPE *in_b, DATA_ITEM_TYPE scalar, DATA_ITEM_TYPE *d_out, int n) {

  DATA_ITEM_TYPE exp_result;
  unsigned errors = 0;
  for (int i = 0; i < n; i++) {
      KERNEL2(exp_result,in_a[i],scalar,in_b[i]);
      DATA_ITEM_TYPE alpha = 2.3;
      for (int k = 0; k < ITERS; k++)
	KERNEL1(alpha,alpha,in_a[i]);
      exp_result = alpha;

      //      exp_result = in_a[i] + scalar * in_b[i];
#ifdef DEBUG
    if (i == 512)
      printf("%3.2f %3.2f\n", exp_result, d_out[i]);
#endif
    DATA_ITEM_TYPE delta = fabs(d_out[i] - exp_result);
    if (delta/exp_result > ERROR_THRESH) {
      errors++;
#ifdef VERBOSE
      printf("%d %f %f\n", i, exp_result, d_out[i]);
#endif
    }
  }
  fprintf(stderr, "%s\n", (errors > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
  return;
}


double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void dlbench_device(DATA_ITEM_TYPE *in_a, DATA_ITEM_TYPE *in_b, DATA_ITEM_TYPE *in_c, DATA_ITEM_TYPE *in_d, DATA_ITEM_TYPE *in_e, DATA_ITEM_TYPE *in_f, DATA_ITEM_TYPE *in_g, DATA_ITEM_TYPE *in_h,
		  DATA_ITEM_TYPE s, DATA_ITEM_TYPE *out, int n) {

  const DATA_ITEM_TYPE scalar = s;

  hc::array<DATA_ITEM_TYPE, 1> d_in_a(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_b(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_c(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_d(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_e(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_f(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_g(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_h(n);

  hc::array<DATA_ITEM_TYPE, 1> d_out(n);

  t_cp_in_b = mysecond();
  hc::copy(in_b, d_in_b);
  t_cp_in_b = 1.0E6 * (mysecond() - t_cp_in_b);

  t_cp_in_a = mysecond();
  hc::copy(in_a, d_in_a);
  t_cp_in_a = 1.0E6 * (mysecond() - t_cp_in_a);

  t_cp_in_c = mysecond();
  hc::copy(in_c, d_in_c);
  t_cp_in_c = 1.0E6 * (mysecond() - t_cp_in_c);

  t_cp_in_d = mysecond();
  hc::copy(in_d, d_in_d);
  t_cp_in_d = 1.0E6 * (mysecond() - t_cp_in_d);

  t_cp_in_e = mysecond();
  hc::copy(in_e, d_in_e);
  t_cp_in_e = 1.0E6 * (mysecond() - t_cp_in_e);

  t_cp_in_f = mysecond();
  hc::copy(in_f, d_in_f);
  t_cp_in_f = 1.0E6 * (mysecond() - t_cp_in_f);

  t_cp_in_g = mysecond();
  hc::copy(in_g, d_in_g);
  t_cp_in_g = 1.0E6 * (mysecond() - t_cp_in_g);

  t_cp_in_h = mysecond();
  hc::copy(in_h, d_in_h);
  t_cp_in_h = 1.0E6 * (mysecond() - t_cp_in_h);
  

  t_par_loop = mysecond();
  auto future = hc::parallel_for_each(hc::extent<1>(n), [&,scalar](hc::index<1> index) [[hc]] {
      d_out[index] = d_in_a[index] +  scalar * d_in_b[index] + d_in_c[index] + d_in_d[index]
      + d_in_e[index] + d_in_f[index] + d_in_g[index] + d_in_h[index];
      // KERNEL2(d_out[index],d_in_a[index],scalar,d_in_b[index]);
      DATA_ITEM_TYPE alpha = 2.3;
      for (int k = 0; k < ITERS; k++)
      	KERNEL1(alpha,alpha,d_in_a[index]);
      d_out[index] = alpha;
    });
  future.wait();
  t_par_loop = 1.0E6 * (mysecond() - t_par_loop);

  t_cp_out = mysecond();
  hc::copy(d_out, out);
  t_cp_out = 1.0E6 * (mysecond() - t_cp_out);
}

// void dlbench_device_aos(pixel *src, pixel *dst, int n) {
// }
void dlbench_device(DATA_ITEM_TYPE *in_a, DATA_ITEM_TYPE *in_b, DATA_ITEM_TYPE *in_c, DATA_ITEM_TYPE *out, int n) {

  hc::array<DATA_ITEM_TYPE, 1> d_in_a(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_b(n);
  hc::array<DATA_ITEM_TYPE, 1> d_in_c(n);

  hc::array<DATA_ITEM_TYPE, 1> d_out(n);

  t_cp_in_a = mysecond();
  hc::copy(in_a, d_in_a);
  t_cp_in_a = 1.0E6 * (mysecond() - t_cp_in_a);

  t_cp_in_b = mysecond();
  hc::copy(in_b, d_in_b);
  t_cp_in_b = 1.0E6 * (mysecond() - t_cp_in_b);

  t_cp_in_c = mysecond();
  hc::copy(in_c, d_in_c);
  t_cp_in_c = 1.0E6 * (mysecond() - t_cp_in_c);

  t_par_loop = mysecond();
  hc::extent<1> ex(THREADS);
  hc::tiled_extent<1> tiled_ex = ex.tile(WKGRP);
  auto future = hc::parallel_for_each(tiled_ex, [&](hc::index<1> index) [[hc]] {
      for (int j = 0; j < SIZE * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
	DATA_ITEM_TYPE alpha = 2.3;
	KERNEL2(alpha,d_in_a[index + j],d_in_b[index + j],d_in_c[index + j]);
	for (int k = 0; k < ITERS; k++)
	  KERNEL1(alpha,alpha,d_in_a[index + j]);
	d_out[index + j] = alpha;
      }
      });
  future.wait();
  t_par_loop = 1.0E6 * (mysecond() - t_par_loop);
  
  t_cp_out = mysecond();
  hc::copy(d_out, out);
  t_cp_out = 1.0E6 * (mysecond() - t_cp_out);
}

void dlbench_host(DATA_ITEM_TYPE *in_a, DATA_ITEM_TYPE *in_b, DATA_ITEM_TYPE *in_c, DATA_ITEM_TYPE *out, int n) {

  //  const DATA_ITEM_TYPE scalar = s;
  //  hc::array_view<DATA_ITEM_TYPE, 1> av_in_a(n, in_a);
  //  hc::array_view<DATA_ITEM_TYPE, 1> av_in_b(n, in_b);
  //  hc::array_view<DATA_ITEM_TYPE, 1> av_in_c(n, in_c);
  //  hc::array_view<DATA_ITEM_TYPE, 1> av_out(n, out);

  //  DATA_ITEM_TYPE *av_in_a = in_a;
  //  hc::array<DATA_ITEM_TYPE, 1> av_in_a(n);
  //  hc::array<DATA_ITEM_TYPE, 1> av_in_b(n);

  hc::array<DATA_ITEM_TYPE, 1> av_out(n);

  // t_cp_in_a = mysecond();
  // hc::copy(in_a, av_in_a);
  // t_cp_in_a = 1.0E6 * (mysecond() - t_cp_in_a);

  // t_cp_in_b = mysecond();
  // hc::copy(in_b, av_in_b);
  // t_cp_in_b = 1.0E6 * (mysecond() - t_cp_in_b);

  t_par_loop = mysecond();
  //  auto future = hc::parallel_for_each(hc::extent<1>(n), [=,&av_in_a,&av_in_b](hc::index<1> index) [[hc]] {
  hc::extent<1> ex(THREADS);
  hc::tiled_extent<1> tiled_ex = ex.tile(WKGRP);
  //  auto future = hc::parallel_for_each(tiled_ex, [=](hc::index<1> index) [[hc]] {
  auto future = hc::parallel_for_each(tiled_ex, [=,&av_out](hc::index<1> index) [[hc]] {
      for (int j = 0; j < SIZE * PIXELS_PER_IMG; j = j + PIXELS_PER_IMG) {
	DATA_ITEM_TYPE alpha = 2.3;
	KERNEL2(alpha,in_a[index[0] + j],in_b[index[0] + j],in_c[index[0] + j]);
	for (int k = 0; k < ITERS; k++)
	  KERNEL1(alpha,alpha,in_a[index[0] + j]);
	av_out[index + j] = alpha;
	//      av_out[index] = av_in_a[index] +  scalar * av_in_b[index];
      }
    });
  future.wait();
  t_par_loop = 1.0E6 * (mysecond() - t_par_loop);

  t_cp_out = mysecond();
  hc::copy(av_out, out);
  t_cp_out = 1.0E6 * (mysecond() - t_cp_out);
}



int main() {


  // Set device
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  auto current = accs.at(VEGA);

  hc::accelerator::set_default(current.get_device_path());
#ifdef DEBUG
  std::cout << "Using HC device " << getDeviceName(current) << std::endl;
#endif

  unsigned n = SIZE * PIXELS_PER_IMG;
  unsigned array_size = n * sizeof(DATA_ITEM_TYPE);

  // DATA_ITEM_TYPE *in_c = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_d = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_e = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_f = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_g = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_h = (DATA_ITEM_TYPE *) malloc(array_size);

  double t_alloc = mysecond();
  DATA_ITEM_TYPE *in_a = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostCoherent);
  DATA_ITEM_TYPE *in_b = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostCoherent);
  DATA_ITEM_TYPE *in_c = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostCoherent);
  //  DATA_ITEM_TYPE *in_a = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_b = (DATA_ITEM_TYPE *) malloc(array_size);
  // DATA_ITEM_TYPE *in_c = (DATA_ITEM_TYPE *) malloc(array_size);
  DATA_ITEM_TYPE *out = (DATA_ITEM_TYPE *) malloc(array_size);
  t_alloc = 1.0E6 * (mysecond() - t_alloc);

  // double t_alloc = mysecond();
  // DATA_ITEM_TYPE *in_a = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostPinned);
  // DATA_ITEM_TYPE *in_b = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostPinned);
  // DATA_ITEM_TYPE *in_c = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostPinned);
  // DATA_ITEM_TYPE *out = (DATA_ITEM_TYPE *) am_alloc(array_size, current, amHostPinned);
  // t_alloc = 1.0E6 * (mysecond() - t_alloc);

  DATA_ITEM_TYPE scalar = 0.5;
  
  double t_init = mysecond();
  for (int i = 0; i < n; i++) {
    in_a[i] = 17;
    in_b[i] = 23;
    in_c[i] = 23;
    // in_d[i] = 27;
    // in_e[i] = 29;
    // in_f[i] = 31;
    // in_g[i] = 33;
    // in_h[i] = 35;
  }
  t_init = 1.0E6 * (mysecond() - t_init);

  double t_init_out = mysecond();
  for (int i = 0; i < n; i++) {
     out[i] = 0;
  }
  t_init_out = 1.0E6 * (mysecond() - t_init_out);
  t_kernel = mysecond();
#ifdef HOST
  dlbench_host(in_a, in_b, in_c, out, n);
  //  dlbench_host(in_a, in_b, in_c, in_d, in_e, in_f, in_g, in_h, scalar, out, n);
#else 
  //  dlbench_device(in_a, in_b, in_c, in_d, in_e, in_f, in_g, in_h, scalar, out, n);
  dlbench_device(in_a, in_b, in_c, out, n);
#endif
   t_kernel = 1.0E6 * (mysecond() - t_kernel);
   //   check_results(in_a, in_b, scalar, out, n);
#ifdef VERBOSE  
  fprintf(stdout, "Kernel Function = %3.8f ms\n", t_kernel/1000); 
  fprintf(stdout, "Kernel Loop = %3.8f ms\n", t_par_loop/1000); 
#ifdef DEV
    fprintf(stdout, "Copy in = %3.8f ms\n", (t_cp_in_a + t_cp_in_b)/1000); 
    fprintf(stdout, "Copy out = %3.8f ms\n", t_cp_out/1000); 
    fprintf(stdout, "H2D bandwidth = %3.8f GB/s\n", ((array_size * 2)/1.0E9)/((t_cp_in_a + t_cp_in_b)/1.0E6));
    fprintf(stdout, "H2D bandwidth (A) = %3.8f GB/s\n", ((array_size)/1.0E9)/(t_cp_in_a/1.0E6));
    fprintf(stdout, "H2D bandwidth (B) = %3.8f GB/s\n", ((array_size)/1.0E9)/(t_cp_in_b/1.0E6));
    fprintf(stdout, "D2H bandwidth = %3.8f GB/s\n", (array_size/1.0E9)/((t_cp_out)/1.0E6));
#endif
#else
    fprintf(stdout, "%3.8f,%3.8f", t_kernel/1000, t_par_loop/1000);  
    //    fprintf(stdout, ",%3.8f,%3.8f", t_alloc/1000,t_init/1000);
#ifdef DEV
  fprintf(stdout, ",%3.8f", t_cp_in_a/1000 + t_cp_in_b/1000 + t_cp_in_b/1000); 
  //  fprintf(stdout, ",%3.8f,%3.8f,%3.8f", t_cp_in_a/1000,t_cp_in_b/1000,t_cp_out/1000); 
   // fprintf(stdout, ",%3.8f", ((array_size)/1.0E9)/((t_init/2 + t_alloc/3 + t_cp_in_a)/1.0E6));
   // fprintf(stdout, ",%3.8f", ((array_size)/1.0E9)/((t_init/2 + t_alloc/3 + t_cp_in_b)/1.0E6));
   // fprintf(stdout, ",%3.8f", (array_size/1.0E9)/((t_init_out + t_alloc/3 + t_cp_out)/1.0E6));
  fprintf(stdout, ",%3.8f", ((array_size)/1.0E9)/((t_cp_in_a)/1.0E6));
  fprintf(stdout, ",%3.8f", ((array_size)/1.0E9)/((t_cp_in_b)/1.0E6));
  fprintf(stdout, ",%3.8f", ((array_size)/1.0E9)/((t_cp_in_c)/1.0E6));
  fprintf(stdout, ",%3.8f", (array_size/1.0E9)/((t_cp_out)/1.0E6));
#endif
  fprintf(stdout, "\n");
#endif
  return 0;
}


 
