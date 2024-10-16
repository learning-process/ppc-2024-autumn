// Copyright 2024 Nesterov Alexander
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/kabalova_v_count_symbols/include/count_symbols_mpi.hpp"

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
    if (isalpha(str[i])) {
      result++;
    }
  }
  return result;
}

bool kabalova_v_count_symbols_mpi::Task1Seq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_mpi::Task1Seq::validation() {
  internal_order_test();
  // �� ����� �������� 1 ������, �� ������ ������ 1 ����� - ����� ��������� �������� � ������.
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  // ��� ������ ������ char'��?
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kabalova_v_count_symbols_mpi::Task1Seq::run() {
  internal_order_test();
  result = countSymbols(input_);
  return true;
}

bool  kabalova_v_count_symbols_mpi::Task1Seq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}


bool kabalova_v_count_symbols_mpi::Task1Mpi::pre_processing() {
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
    std::cout << std::endl << input_ << " " << input_.size() << std::endl;
    for (int proc = 1; proc < world.size(); proc++) {
      // input_size() / world.size() not always an integer
      // so the last process sometimes gets memory access violation
      // calculate this "delta" between input_.size() and proc * delta
      int bufDelta = 0;
      if ((proc * delta + delta) > input_.size()) {
        bufDelta = input_.size() - proc * delta - delta;
      }
      std::cout << "bufDelta = " << bufDelta << std::endl;
      world.send(proc, 0, input_.data() + proc * delta, delta + bufDelta);
    }
  }
  // Initialize substring in root
  if (world.rank() == 0)
    local_input_ = input_.substr(0, delta);
  else {
    std::string buffer;
    buffer.resize(delta);
    std::cout << std::endl << "delta = "<< delta<< std::endl;
    // Other processes get substrings from root
    world.recv(0, 0, buffer.data(), delta);
    local_input_ = std::string(buffer.data(), delta);
  }
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_mpi::Task1Mpi::validation() {
  internal_order_test();
  if (world.rank() == 0 ) {
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

bool kabalova_v_count_symbols_mpi::Task1Mpi::run() {
  internal_order_test();
  int local_result = 0;
  // Count symbols in every substring
  local_result = countSymbols(local_input_);
  std::cout << world.rank() << ": " << local_input_ << ": " << local_result << " " << std::endl;
  // Get sum and send it into result
  reduce(world, local_result, result, std::plus(), 0);
  return true;
}

bool kabalova_v_count_symbols_mpi::Task1Mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

