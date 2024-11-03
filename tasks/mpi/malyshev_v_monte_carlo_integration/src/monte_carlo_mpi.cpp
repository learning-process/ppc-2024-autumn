#include "mpi/malyshev_v_monte_carlo_integration/include/monte_carlo_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <numeric>
#include <random>

namespace malyshev_v_mci_mpi {

void MonteCarloMPITask::set_function(std::function<double(double)> func, double a, double b, int num_samples) {
  func_ = std::move(func);
  a_ = a;
  b_ = b;
  num_samples_ = num_samples;
}

bool MonteCarloMPITask::pre_processing() {
  if (world.rank() == 0) {
    // Настройка границ интегрирования и количества выборок
    broadcast(world, a_, 0);
    broadcast(world, b_, 0);
    broadcast(world, num_samples_, 0);
  } else {
    // Инициализация переменных на каждом процессе
    broadcast(world, a_, 0);
    broadcast(world, b_, 0);
    broadcast(world, num_samples_, 0);
  }
  return true;
}

bool MonteCarloMPITask::validation() { return taskData->outputs_count[0] == 1; }

bool MonteCarloMPITask::run() {
  local_result_ = parallel_integration();
  reduce(world, local_result_, global_result_, std::plus<>(), 0);
  return true;
}

bool MonteCarloMPITask::post_processing() {
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result_;
  }
  return true;
}

double MonteCarloMPITask::parallel_integration() {
  int samples_per_process = num_samples_ / size;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(a_, b_);

  double local_sum = 0.0;
  for (int i = 0; i < samples_per_process; ++i) {
    double x = dist(gen);
    local_sum += func_(x);
  }

  return (b_ - a_) * local_sum / samples_per_process;
}

}  // namespace malyshev_v_mci_mpi
