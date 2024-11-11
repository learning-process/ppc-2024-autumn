#include "mpi/polikanov_v_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace std::chrono_literals;

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int count = static_cast<int>(taskData->inputs_count[0]);
  int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(count);
  std::copy(input, input + count, input_.begin());
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0) || taskData->outputs_count[0] == 1;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::run() {
  internal_order_test();
  int count = input_.size();
  for (int i = 0; i < count; i++) {
    res = std::max(res, input_[i]);
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  }
  res = 0;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  res = 0;
  int max = INT_MIN;
  for (size_t i = 0; i < local_input_.size(); ++i) {
    max = std::max(max, local_input_[i]);
  }
  reduce(world, max, res, boost::mpi::maximum<int>(), 0);
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
