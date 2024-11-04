// Copyright 2024 Chastov Vyacheslav
#include "mpi/chastov_v_count_words_in_line/include/ops_mpi.hpp"

namespace chastov_v_count_words_in_line_mpi {

std::vector<char> createString(int n) {
  std::vector<char> wordCountInput;
  std::string testString = "This is a proposal to evaluate the \n performance of a word counting algorithm via MPI. ";
  for (int i = 0; i < n; i++) {
    for (unsigned long int j = 0; j < testString.length(); j++) {
      wordCountInput.push_back(testString[j]);
    }
  }
  return wordCountInput;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* temp = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = temp[i];
  }
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  for (char c : input_) {
    if (c == ' ') {
      spacesFound++;
    }
  }
  wordsFound = spacesFound + 1;
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordsFound;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int blockSize = 0;
  if (world.rank() == 0) {
    input_ = std ::vector<char>(taskData->inputs_count[0]);
    auto* tmp = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned long int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp[i];
    }
    blockSize = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, blockSize, 0);

  local_input_.resize(blockSize);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * blockSize, blockSize);
    }
    local_input_ = std::vector<char>(input_.begin(), input_.begin() + blockSize);
  } else {
    world.recv(0, 0, local_input_.data(), blockSize);
  }
  for (char c : local_input_) {
    if (c == ' ') {
      localSpaceFound++;
    }
  }
  boost::mpi::reduce(world, localSpaceFound, spacesFound, std::plus<>(), 0);
  if (world.rank() == 0) {
    wordsFound = spacesFound + 1;
  }
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = wordsFound;
  }
  return true;
}

}  // namespace chastov_v_count_words_in_line_mpi