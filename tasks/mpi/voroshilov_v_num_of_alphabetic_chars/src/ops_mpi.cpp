#include "mpi/voroshilov_v_num_of_alphabetic_chars/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<char>(taskData->inputs_count[0]);
  char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    if (std::isalpha(input_[i]) != 0) {  // Check if it is alphabetic character
      res_++;
    }
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of input and output
    return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::pre_processing() {
  internal_order_test();
  // Init value for output
  res_ = 0;
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::run() {
  internal_order_test();

  std::vector<char> input_;
  size_t part = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    part = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }
  boost::mpi::broadcast(world, part, 0);
  boost::mpi::broadcast(world, remainder, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<char>(taskData->inputs_count[0]);
    char* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    std::copy(ptr, ptr + taskData->inputs_count[0], input_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + remainder + proc * part, part);
    }
  }

  local_input_ = std::vector<char>(part);
  if (world.rank() == 0) {
    local_input_ = std::vector<char>(input_.begin(), input_.begin() + remainder + part);
  } else {
    world.recv(0, 0, local_input_.data(), part);
  }

  int local_res = 0;
  for (size_t i = 0; i < local_input_.size(); i++) {
    if (std::isalpha(local_input_[i]) != 0) {  // Check if it is alphabetic character
      local_res++;
    }
  }
  boost::mpi::reduce(world, local_res, res_, std::plus(), 0);
  return true;
}

bool voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res_;
  }
  return true;
}
