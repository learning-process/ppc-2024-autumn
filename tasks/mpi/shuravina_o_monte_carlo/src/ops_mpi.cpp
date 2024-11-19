#include "mpi/shuravina_o_monte_carlo/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
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
      std::cerr << "Validation failed: inputs_count or outputs_count size mismatch" << std::endl;
      return false;
    }
    if (taskData->inputs_count[0] != 0 || taskData->outputs_count[0] != 1) {
      std::cerr << "Validation failed: inputs_count or outputs_count value mismatch" << std::endl;
      return false;
    }
    if (taskData->inputs[0] != nullptr || taskData->outputs[0] == nullptr) {
      std::cerr << "Validation failed: inputs or outputs pointer mismatch" << std::endl;
      return false;
    }
  }

  return true;
}
bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::run() {
  internal_order_test();
  int num_processes = world.size();
  int rank = world.rank();

  std::cout << "Run proc " << world.rank() << "\n";
  int local_num_points = num_points_ / num_processes;

  double local_sum = 0.0;
  for (int i = 0; i < local_num_points; ++i) {
    double x = dis_(gen);
    local_sum += f_(x);
  }

  double global_sum = 0.0;
  boost::mpi::all_reduce(world, local_sum, global_sum, std::plus<>());

  integral_value_ = (global_sum / num_points_) * (b_ - a_);

  std::cout << "Rank " << rank << " finished processing." << std::endl;

  return true;
}
bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = integral_value_;
  }
  return true;
}