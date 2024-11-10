#include "mpi/shuravina_o_monte_carlo/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <iostream>
#include <random>

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::pre_processing() {
  internal_order_test();
  integral_value_ = 0.0;
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
      return false;
    }
    if (taskData->inputs_count[0] != 0 || taskData->outputs_count[0] != 1) {
      return false;
    }
    if (taskData->inputs[0] != nullptr || taskData->outputs[0] == nullptr) {
      return false;
    }
  }
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::run() {
  internal_order_test();
  int num_processes = world.size();
  int rank = world.rank();

  std::cout << "Rank " << rank << " is running." << std::endl;

  int local_num_points = num_points_ / num_processes;
  int remainder = num_points_ % num_processes;

  if (rank < remainder) {
    local_num_points++;
  }

  std::cout << "Rank " << rank << " has " << local_num_points << " points." << std::endl;

  double local_sum = 0.0;
  for (int i = 0; i < local_num_points; ++i) {
    double x = dis_(gen);
    local_sum += f_(x);
  }

  std::cout << "Rank " << rank << " local sum: " << local_sum << std::endl;

  double global_sum = 0.0;
  boost::mpi::all_reduce(world, local_sum, global_sum, std::plus<>());

  integral_value_ = (global_sum / num_points_) * (b_ - a_);

  std::cout << "Rank " << rank << " global sum: " << global_sum << std::endl;

  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = integral_value_;
  }
  return true;
}