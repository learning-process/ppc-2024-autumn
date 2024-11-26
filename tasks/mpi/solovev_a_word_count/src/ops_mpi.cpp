#include "mpi/solovev_a_word_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace solovev_a_word_count_mpi {

bool solovev_a_word_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  res = 0;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (char symbol : input_) {
    if (symbol == ' ' || symbol == '.') {
      res++;
    }
  }
  return true;
}
bool solovev_a_word_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std ::vector<char>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }
  res = 0;
  l_res = 0;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1);
  }
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, delta, 0);
  if (world.rank() == 0) {
    for (int p = 1; p < world.size(); p++) {
      world.send(p, 0, input_.data() + p * delta, delta);
    }
  }
  l_input_.resize(delta);
  if (world.rank() == 0) {
    l_input_ = std::vector<char>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, l_input_.data(), delta);
  }
  for (char symbol : input_) {
    if (symbol == ' ' || symbol == '.') {
      l_res++;
    }
  }
  boost::mpi::reduce(world, l_res, res, std::plus<>(), 0);
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

}  // namespace solovev_a_word_count_mpi
