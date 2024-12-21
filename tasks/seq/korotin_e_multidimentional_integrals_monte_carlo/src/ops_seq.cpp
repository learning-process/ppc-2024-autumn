// Copyright 2024 Nesterov Alexander
#include "seq/korotin_e_multidimentional_integrals_monte_carlo/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <thread>

using namespace std::chrono_literals;

bool korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  dim = taskData->inputs_count[1];
  N = (reinterpret_cast<size_t*>(taskData->inputs[2]))[0];
  input_ = std::vector<std::pair<double, double>>(dim);
  auto* start = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[1]);
  f = (reinterpret_cast<double (**)(double*, int)>(taskData->inputs[0]))[0];
  std::copy(start, start + dim, input_.begin());
  // Init value for output
  res = 0.0;
  M = 0.0;
  variance = -1.0;
  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[2] == 1 && taskData->inputs_count[0] == 1;
}

bool korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential::run() {
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

  delete[] mas;
  delete[] rng_bord;

  return true;
}

bool korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

double korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential::possible_error() {
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
  return 6 * std::abs(volume) * sqrt(variance);
}
