#include "mpi/guseynov_e_check_lex_order_of_two_string/include/ops_mpi.hpp"

#include <vector>

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // init vectors
  input_ = std::vector<std::vector<char>>(taskData->inputs_count[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    input_[i] = std::vector<char>(taskData->inputs_count[i + 1]);
    for (unsigned j = 0; j < taskData->inputs_count[i + 1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res_ = 0;
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // check count of words and count of elements of output
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < std::min(input_[0].size(), input_[1].size()); i++) {
    if (input_[0][i] < input_[1][i]) {
      res_ = 1;
      break;
    }
    if (input_[0][i] > input_[1][i]) {
      res_ = 2;
      break;
    }
  }
  if (res_ == 0 && input_[0].size() != input_[1].size()) {
    if (input_[0].size() > input_[1].size()) {
      res_ = 2;
    } else {
      res_ = 1;
    }
  }
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = std::min(taskData->inputs_count[1], taskData->inputs_count[2]) / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // init vectors
    input_ = std::vector<std::vector<char>>(taskData->inputs_count[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
      input_[i] = std::vector<char>(taskData->inputs_count[i + 1]);
      for (unsigned j = 0; j < taskData->inputs_count[i + 1]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_[0].data() + delta * proc, delta);
      world.send(proc, 1, input_[1].data() + delta * proc, delta);
    }
  }
  local_input_1_ = std::vector<char>(delta);
  local_input_2_ = std::vector<char>(delta);
  if (world.rank() == 0) {
    local_input_1_ = std::vector<char>(input_[0].begin(), input_[0].begin() + delta);
    local_input_2_ = std::vector<char>(input_[1].begin(), input_[1].begin() + delta);
  } else {
    world.recv(0, 0, local_input_1_.data(), delta);
    world.recv(0, 1, local_input_2_.data(), delta);
  }
  // Init value for output
  res_ = 0;
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  for (size_t i = 0; i < local_input_1_.size(); i++) {
    if (local_input_1_[i] < local_input_2_[i]) {
      local_res = 1;
      break;
    }
    if (local_input_1_[i] > local_input_2_[i]) {
      local_res = 2;
      break;
    }
  }

  std::vector<int> gathered_data;
  boost::mpi::gather(world, local_res, gathered_data, 0);

  if (world.rank() == 0) {
    for (int proc = 0; proc < world.size(); proc++) {
      if (gathered_data[proc] != 0) {
        res_ = gathered_data[proc];
        break;
      }
    }
    if (res_ == 0 && input_[0].size() != input_[1].size()) {
      if (input_[0].size() > input_[1].size()) {
        res_ = 2;
      } else {
        res_ = 1;
      }
    }
  }
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}