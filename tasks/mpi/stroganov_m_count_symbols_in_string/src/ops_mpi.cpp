// Copyright 2024 Stroganov Mikhail
#include "mpi/stroganov_m_count_symbols_in_string/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int stroganov_m_count_symbols_in_string_mpi::getRandomNum(int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return ((gen() % (max - min + 1)) + min);
}

std::string stroganov_m_count_symbols_in_string_mpi::getRandomString() {
  std::string result;
  std::string dictionary = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz1234567890";
  int str_len = getRandomNum(1000, 20000);
  for (int i = 0; i < str_len; i++) {
    result += dictionary[getRandomNum(0, dictionary.size() - 1)];
  }
  return result;
}

int stroganov_m_count_symbols_in_string_mpi::countOfSymbols(std::string& str) {
  int result = 0;
  size_t n = str.size();
  for (size_t i = 0; i < n; i++) {
    if (isalpha(str[i]) != 0) {
      result++;
    }
  }
  return result;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  bool valid_len = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  bool is_char_array = typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name();
  return valid_len && is_char_array;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  result = countOfSymbols(input_);
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int partition_size = 0;
  if (world.rank() == 0) {
    partition_size = (taskData->inputs_count[0] + world.size() - 1) / world.size();
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    for (int proc = 1; proc < world.size(); ++proc) {
      unsigned int start_idx = proc * partition_size;
      unsigned int size_to_send =
          (start_idx + partition_size > input_.size()) ? input_.size() - start_idx : partition_size;
      world.send(proc, 0, size_to_send);
      world.send(proc, 0, input_.data() + start_idx, size_to_send);
    }
    local_input_ = input_.substr(0, partition_size);
  } else {
    unsigned int received_size = 0;
    world.recv(0, 0, received_size);
    std::vector<char> buffer(received_size);
    world.recv(0, 0, buffer.data(), received_size);
    local_input_ = std::string(buffer.data(), buffer.size());
  }
  result = 0;
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    bool valid_input = taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
    bool is_char_array = typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name();
    return valid_input && is_char_array;
  }
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_result = 0;
  local_result = countOfSymbols(local_input_);
  reduce(world, local_result, result, std::plus(), 0);
  return true;
}

bool stroganov_m_count_symbols_in_string_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  }
  return true;
}
