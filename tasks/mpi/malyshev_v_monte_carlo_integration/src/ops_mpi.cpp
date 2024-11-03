#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <random>

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::pre_processing() {
  internal_order_test();

  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* tmp_ptr_n = reinterpret_cast<int*>(taskData->inputs[2]);

  a_ = *tmp_ptr_a;
  b_ = *tmp_ptr_b;
  n_ = *tmp_ptr_n;

  return true;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::run() {
  internal_order_test();
  result_ = integrate(func_, a_, b_, n_);
  return true;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = result_;
  return true;
}

double malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::integrate(
    const std::function<double(double)>& f, double a, double b, int n) {
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist(a, b);

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double x = dist(rng);
    sum += f(x);
  }

  return (b - a) * sum / n;
}

void malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::pre_processing() {
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

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::run() {
  internal_order_test();
  MPI_Bcast(&a_, sizeof(a_) + sizeof(b_) + sizeof(n_), MPI_BYTE, 0, world);

  double local_result = parallel_integrate(func_, a_, b_, n_);
  reduce(world, local_result, global_result_, std::plus<>(), 0);

  return true;
}

bool malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result_;
  }
  return true;
}

double malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::parallel_integrate(
    const std::function<double(double)>& f, double a, double b, int n) {
  int rank = world.rank();
  int size = world.size();

  std::mt19937 rng(rank); // Seed with rank to ensure different streams
  std::uniform_real_distribution<double> dist(a, b);

  double local_sum = 0.0;
  int local_n = n / size;

  for (int i = 0; i < local_n; ++i) {
    double x = dist(rng);
    local_sum += f(x);
  }

  return (b - a) * local_sum / n;
}

void malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}
