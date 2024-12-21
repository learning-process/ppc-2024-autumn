#include "mpi/korotin_e_multidimentional_integrals_monte_carlo/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  dim = taskData->inputs_count[1];
  N = (reinterpret_cast<size_t*>(taskData->inputs[2]))[0];
  input_ = std::vector<std::pair<double, double>>(dim);
  auto* start = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[1]);
  std::copy(start, start + dim, input_.begin());
  f = (reinterpret_cast<double (**)(double*, int)>(taskData->inputs[0]))[0];
  // Init value for output
  res = 0.0;
  M = 0.0;
  variance = -1.0;
  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] == 1 && taskData->inputs_count[2] == 1;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  std::uniform_real_distribution<double>* rng_bord = new std::uniform_real_distribution<double>[dim];
  double* mas = new double[dim];

  for (int i = 0; i < dim; i++) {
    if (input_[i].first > input_[i].second)
      rng_bord[i] = std::uniform_real_distribution<double>(input_[i].second, input_[i].first);
    else
      rng_bord[i] = std::uniform_real_distribution<double>(input_[i].first, input_[i].second);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  rng = std::vector<double>(N);
  for (size_t i = 0; i < N; i++) {
    for (int j = 0; j < dim; j++) mas[j] = rng_bord[j](gen);
    rng[i] = f(mas, dim) / N;
  }
  M = std::accumulate(rng.begin(), rng.end(), M);

  double volume = 1.0;
  for (int i = 0; i < dim; i++) volume *= (input_[i].second - input_[i].first);
  res = volume * M;

  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

double korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskSequential::possible_error() {
  double volume = 1.0;
  for (int i = 0; i < dim; i++) volume *= (input_[i].second - input_[i].first);

  if (variance < 0) {
    if (rng.size() == N) {
      M *= (-M / N);
      for (size_t i = 0; i < N; i++) {
        rng[i] *= rng[i];
      }
      variance = std::accumulate(rng.begin(), rng.end(), M);
    } else
      return -1.0;
  }
  return 6 * std::abs(volume) * variance;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    dim = taskData->inputs_count[1];
    N = (reinterpret_cast<size_t*>(taskData->inputs[3]))[0];
    input_left_ = std::vector<double>(dim);
    input_right_ = std::vector<double>(dim);
    auto* start1 = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* start2 = reinterpret_cast<double*>(taskData->inputs[2]);
    std::copy(start1, start1 + dim, input_left_.begin());
    std::copy(start2, start2 + dim, input_right_.begin());
  }
  f = (reinterpret_cast<double (**)(double*, int)>(taskData->inputs[0])[0]);
  res = 0.0;
  M = 0.0;
  local_M = 0.0;
  variance = -1.0;
  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[1] == taskData->inputs_count[2] &&
           taskData->inputs_count[3] == 1 && taskData->inputs_count[0] == 1;
  }
  return taskData->inputs_count[0] == 1;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, dim, 0);
  broadcast(world, N, 0);
  broadcast(world, input_left_, 0);
  broadcast(world, input_right_, 0);

  std::uniform_real_distribution<double>* rng_bord = new std::uniform_real_distribution<double>[dim];
  double* mas = new double[dim];
  if (world.rank() < static_cast<int>(N % world.size()))
    n = N / world.size() + 1;
  else
    n = N / world.size();

  for (int i = 0; i < dim; i++) {
    if (input_left_[i] > input_right_[i])
      rng_bord[i] = std::uniform_real_distribution<double>(input_right_[i], input_left_[i]);
    else
      rng_bord[i] = std::uniform_real_distribution<double>(input_left_[i], input_right_[i]);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  rng = std::vector<double>(n);
  for (size_t i = 0; i < n; i++) {
    for (int j = 0; j < dim; j++) mas[j] = rng_bord[j](gen);
    rng[i] = f(mas, dim) / N;
  }
  local_M = std::accumulate(rng.begin(), rng.end(), local_M);

  reduce(world, local_M, M, std::plus(), 0);

  if (world.rank() == 0) {
    double volume = 1.0;
    for (int i = 0; i < dim; i++) volume *= (input_right_[i] - input_left_[i]);
    res = volume * M;
  }

  delete[] mas;
  delete[] rng_bord;

  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

double korotin_e_multidimentional_integrals_monte_carlo_mpi::TestMPITaskParallel::possible_error() {
  local_variance = 0.0;
  if (variance < 0) {
    if (rng.size() == n) {
      for (size_t i = 0; i < n; i++) {
        rng[i] *= rng[i];
      }
      local_variance = std::accumulate(rng.begin(), rng.end(), local_variance);
    } else
      return -1.0;
  }

  reduce(world, local_variance, variance, std::plus(), 0);

  double volume = 1.0;
  for (int i = 0; i < dim; i++) volume *= (input_right_[i] - input_left_[i]);

  if (world.rank() == 0) {
    M *= (M / N);
    variance -= M;
  }

  broadcast(world, variance, 0);

  return 6 * std::abs(volume) * sqrt(variance);
}
