#include "mpi/nikolaev_r_trapezoidal_integral/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::pre_processing() {
  internal_order_test();
  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);
  a_ = *tmp_ptr_a, b_ = *tmp_ptr_b, n_ = *tmp_ptr_n;
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::run() {
  internal_order_test();
  res_ = integrate_function(a_, b_, n_, function_);
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = res_;
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);

    a_ = *tmp_ptr_a;
    b_ = *tmp_ptr_b;
    n_ = static_cast<int>(*tmp_ptr_n);
  }
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::run() {
  internal_order_test();
  double params[3] = {0.0};
  if (world.rank() == 0) {
    params[0] = a_;
    params[1] = b_;
    params[2] = static_cast<double>(n_);
  }
  MPI_Bcast(&params, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double local_res{};
  local_res = integrate_function(a_, b_, n_, function_);
  MPI_Reduce(&local_res, &res_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  return true;
}

bool nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res_;
  }
  return true;
}

void nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::set_function(
    const std::function<double(double)>& f) {
  function_ = f;
}

void nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::set_function(
    const std::function<double(double)>& f) {
  function_ = f;
}

double nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralSequential::integrate_function(
    double a, double b, int n, const std::function<double(double)>& f) {
  const double width = (b - a) / n;
  double result = 0.0;
  for (int step = 0; step < n; step++) {
    const double x1 = a + step * width;
    const double x2 = a + (step + 1) * width;

    result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
  }

  return result;
}

double nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel::integrate_function(
    double a, double b, int n, const std::function<double(double)>& f) {
  int rank = world.rank();
  int size = world.size();

  const double width = (b - a) / n;
  double result = 0.0;
  for (int step = rank; step < n; step += size) {
    const double x1 = a + step * width;
    const double x2 = a + (step + 1) * width;

    result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
  }

  return result;
}