#include "mpi/gusev_n_trapezoidal_rule/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::pre_processing() {
  internal_order_test();

  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);

  a_ = *tmp_ptr_a;
  b_ = *tmp_ptr_b;
  n_ = *tmp_ptr_n;

  return true;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::run() {
  internal_order_test();
  result_ = integrate(func_, a_, b_, n_);
  return true;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = result_;
  return true;
}

double gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::integrate(const std::function<double(double)>& f,
                                                                                 double a, double b, int n) {
  double h = (b - a) / n;
  double sum = 0.5 * (f(a) + f(b));

  for (int i = 1; i < n; ++i) {
    double x = a + i * h;
    sum += f(x);
  }

  return sum * h;
}

void gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);

    a_ = *tmp_ptr_a;
    b_ = *tmp_ptr_b;
    n_ = *tmp_ptr_n;
  }

  return true;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::run() {
  internal_order_test();
  MPI_Bcast(&a_, sizeof(a_) + sizeof(b_) + sizeof(n_), MPI_BYTE, 0, world);
  double local_result = parallel_integrate(func_, a_, b_, n_);
  reduce(world, local_result, global_result_, std::plus<>(), 0);
  return true;
}

bool gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result_;
  }
  return true;
}

double gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::parallel_integrate(
    const std::function<double(double)>& f, double a, double b, int n) {
  int rank = world.rank();
  int size = world.size();

  double h = (b - a) / n;
  double local_sum = 0.0;

  for (int i = rank; i < n; i += size) {
    double x = a + i * h;
    local_sum += f(x);
  }

  if (rank == 0) {
    local_sum += 0.5 * (f(a) + f(b));
  }

  return local_sum * h;
}

void gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}