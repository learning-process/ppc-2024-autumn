#include "mpi/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

double shulpin_monte_carlo_integration::fsin(double x) { return std::sin(x); }
double shulpin_monte_carlo_integration::fcos(double x) { return std::cos(x); }
double shulpin_monte_carlo_integration::f_two_sin_cos(double x) { return 2 * std::sin(x) * std::cos(x); }

double shulpin_monte_carlo_integration::integral(double a, double b, int N, const func& func_seq) {
  double h = (b - a) / (N * 1.0);
  double sum = 0.0;

  for (int i = 0; i < N; ++i) {
    sum += func_seq(a + h * i);
  }

  return h * sum;
}

double shulpin_monte_carlo_integration::parallel_integral(double a, double b, int N, const func& func_MPI) {
  boost::mpi::communicator world;
  double chunk = (b - a) / world.size();

  double a_chunk = a + chunk * world.rank();
  double b_chunk = a_chunk + chunk;
  double h_chunk = (b_chunk - a_chunk) / (N * 1.0);

  double local_sum = 0.0;

  for (int i = 0; i < N; ++i) {
    local_sum += func_MPI(a_chunk + h_chunk * i);
  }
  local_sum *= h_chunk;

  return local_sum;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
  double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
  int N_value = *reinterpret_cast<int*>(taskData->inputs[2]);

  a_seq = a_value;
  b_seq = b_value;
  N_seq = N_value;

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::run() {
  internal_order_test();

  res = integral(a_seq, b_seq, N_seq, func_seq);

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
    double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
    int N_value = *reinterpret_cast<int*>(taskData->inputs[2]);

    a_MPI = a_value;
    b_MPI = b_value;
    N_MPI = N_value;
  }

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->inputs_count.size() == 3 && taskData->outputs_count[0] == 1);
  }

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskParallel::run() {
  internal_order_test();

  double local_res{};

  boost::mpi::broadcast(world, a_MPI, 0);
  boost::mpi::broadcast(world, b_MPI, 0);
  boost::mpi::broadcast(world, N_MPI, 0);

  local_res = parallel_integral(a_MPI, b_MPI, N_MPI, func_MPI);
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }

  return true;
}

void shulpin_monte_carlo_integration::TestMPITaskSequential::set_seq(const func& f) { func_seq = f; }

void shulpin_monte_carlo_integration::TestMPITaskParallel::set_MPI(const func& f) { func_MPI = f; }