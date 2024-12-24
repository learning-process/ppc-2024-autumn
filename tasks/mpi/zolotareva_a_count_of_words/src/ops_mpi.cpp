// Copyright 2023 Nesterov Alexander
// здесь писать саму задачу
#include "mpi/zolotareva_a_count_of_words/include/ops_mpi.hpp"

#include <string>

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  res = 0;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  bool in_word = false;
  for (char c : input_) {
    if (c == ' ' && in_word) {
      ++res;
      in_word = false;
    } else if (c != ' ') {
      in_word = true;
    }
  }
  if (in_word) ++res;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
  }
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int world_size = world.size();
  unsigned int delta;
  if (world.rank() == 0) delta = taskData->inputs_count[0] / world_size;
  if (world_size == 1) {
    local_input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    return true;
  }
  boost::mpi::broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    unsigned int str_size = taskData->inputs_count[0];
    unsigned int remainder = str_size % world_size;
    local_input_ = input_.substr(0, remainder + delta);
    for (int proc = 1; proc < world_size; proc++) {
      world.send(proc, 0, input_.data() + remainder + proc * delta - 1, delta + 1);
    }
  } else {
    local_input_ = std::string(delta + 1, '\0');
    world.recv(0, 0, local_input_.data(), delta + 1);
  }
  res = 0;
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  bool in_word = false;

  for (char c : local_input_) {
    if (c == ' ' && in_word) {
      ++local_res;
      in_word = false;
    } else if (c != ' ') {
      in_word = true;
    }
  }
  if (in_word) ++local_res;
  if (world.rank() != 0 && local_input_[0] != ' ' && in_word) {
    --local_res;
  }
  if (world.rank() == (world.size() - 1) && local_input_[0] != ' ' && !in_word) {
    --local_res;
  }
  boost::mpi::reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool zolotareva_a_count_of_words_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}