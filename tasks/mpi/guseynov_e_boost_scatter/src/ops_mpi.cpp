#include "mpi/guseynov_e_boost_scatter/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>

bool guseynov_e_boost_scatter_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init Vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res_ = 0;
  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res_ = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  res_ = 0;
  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    delta = input_.size() / world.size();
    local_input_ = std::vector<int>(delta + input_.size() % world.size());
  }
  broadcast(world, delta, 0);

  if (world.rank() != 0) {
    local_input_ = std::vector<int>(delta);
  }

  boost::mpi::scatter(world, input_, local_input_.data(), delta, 0);
  if (world.rank() == 0) {
    std::copy(input_.begin() + delta * world.size(), input_.end(), local_input_.begin() + delta);
  }

  int local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res_, std::plus(), 0);

  return true;
}

bool guseynov_e_boost_scatter_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}