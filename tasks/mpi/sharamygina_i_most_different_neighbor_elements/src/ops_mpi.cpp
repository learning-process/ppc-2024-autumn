#include "mpi/sharamygina_i_most_different_neighbor_elements/include/ops_mpi.hpp"

#include <functional>
#include <thread>
#include <vector>

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());

  res = abs(input_[0] - input_[1]);

  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::run() {
  internal_order_test();

  for (size_t i = 2; i < input_.size(); ++i) {
    res = std::max(res, abs(input_[i] - input_[i - 1]));
  }
  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::pre_processing() {
  internal_order_test();

  // Init value for output
  res = INT_MIN;
  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::run() {
  internal_order_test();
  int delta = 0;
  if (world.rank() == 0) {
    delta = (taskData->inputs_count[0]) / world.size();
    size = taskData->inputs_count[0];
    if (taskData->inputs_count[0] % world.size() > 0u) delta++;
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_.resize(world.size() * delta + 2, 0);
    auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp, tmp + taskData->inputs_count[0], input_.begin());

    for (size_t i = taskData->inputs_count[0]; i < input_.size(); i++) {
      input_[i] = input_[i - 1];
    }

    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta + 1);
    }
  }

  local_input_.resize(delta + 1);

  st = world.rank() * delta;
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + 1);
  } else {
    world.recv(0, 0, local_input_.data(), delta + 1);
  }
  int lans = INT_MIN;
  for (size_t i = 0; i < local_input_.size() - 1; ++i) {
    lans = std::max(abs(local_input_[i + 1] - local_input_[i]), lans);
  }
  reduce(world, lans, res, boost::mpi::maximum<int>(), 0);
  return true;
}

bool sharamygina_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
