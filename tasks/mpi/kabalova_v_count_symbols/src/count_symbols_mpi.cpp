// Copyright 2024 Kabalova Valeria
#include "mpi/kabalova_v_count_symbols/include/count_symbols_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int kabalova_v_count_symbols_mpi::getRandomNumber(int left, int right) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return ((gen() % (right - left + 1)) + left);
}

std::string kabalova_v_count_symbols_mpi::getRandomString() {
  std::string str;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz1234567890";
  int strSize = getRandomNumber(1000, 20000);
  for (int i = 0; i < strSize; i++) {
    str += alphabet[getRandomNumber(0, alphabet.size() - 1)];
  }
  return str;
}

int kabalova_v_count_symbols_mpi::countSymbols(std::string& str) {
  int result = 0;
  for (size_t i = 0; i < str.size(); i++) {
    if (isalpha(str[i]) != 0) {
      result++;
    }
  }
  return result;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // На выход подается 1 строка, на выходе только 1 число - число буквенных символов в строке.
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  // Нам пришел массив char'ов?
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kabalova_v_count_symbols_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  result = countSymbols(input_);
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    // Get delta = string.size() / num_threads
    delta = taskData->inputs_count[0] % world.size() == 0 ? taskData->inputs_count[0] / world.size()
                                                          : taskData->inputs_count[0] / world.size() + 1;
  }
  broadcast(world, delta, 0);
  // Initialize main string in root
  // Then send substrings to processes
  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    for (int proc = 1; proc < world.size(); proc++) {
      // input_size() / world.size() not always an integer
      // so the last process sometimes gets memory access violation
      // calculate this "delta" between input_.size() and proc * delta
      // also if number of processes larger than world.size() then bufdelta is zero and they other processes get empty
      // string
      int bufDelta = 0;
      if ((size_t)(proc * delta + delta) > input_.size() && (size_t)proc < input_.size()) {
        bufDelta = input_.size() - proc * delta - delta;
      }
      world.send(proc, 0, input_.data() + proc * delta, delta + bufDelta);
    }
  }
  // Initialize substring in root
  if (world.rank() == 0)
    local_input_ = input_.substr(0, delta);
  else {
    std::string buffer;
    buffer.resize(delta);
    // Other processes get substrings from root
    world.recv(0, 0, buffer.data(), delta);
    local_input_ = std::string(buffer.data(), delta);
  }
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // 1 input string - 1 output number
    bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
    // Did we get array of chars?
    bool flag2 = false;
    if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
      flag2 = true;
    }
    return (flag1 && flag2);
  }
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_result = 0;
  // Count symbols in every substring
  local_result = countSymbols(local_input_);
  // Get sum and send it into result
  reduce(world, local_result, result, std::plus(), 0);
  return true;
}

bool kabalova_v_count_symbols_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}