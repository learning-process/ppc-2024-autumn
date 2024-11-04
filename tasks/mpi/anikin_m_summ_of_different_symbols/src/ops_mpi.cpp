// Copyright 2024 Anikin Maksim
#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// SEQ
bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
  input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
  // Init value for output
  res = 0;
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::run() {
  internal_order_test();
  int dif = 0;
  std::string str1 = input[0];  
  std::string str2 = input[1];
  if (str1.size() >= str2.size()) {
    dif = str1.size() - str2.size();
  } else {
    dif = str2.size() - str1.size();
  }
  auto i1 = str1.begin();
  auto i2 = str2.begin();
  while (i1 != str1.end() && i2 != str2.end()) {
    if(*i1 != *i2) res++;
    i1++;
    i2++;
  }
  res += dif;
  std::cout << res;
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

// MPI
bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    if (strlen(reinterpret_cast<char *>(taskData->inputs[0])) >=
        strlen(reinterpret_cast<char *>(taskData->inputs[1]))) {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
    } else {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    }
    if (strlen(input[0]) != (strlen(input[1]))) {
      res = strlen(input[0]) - strlen(input[1]);
      input[0][strlen(input[1])] = '\0';
    } else {
      res = 0;
    }
  }
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::validation() {
  internal_order_test();
  if (com.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] == 2;
  }
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::run() {
  internal_order_test();
  std::string str1 = input[0];
  std::string str2 = input[1];
  int str_len = str1.size();

  int local_count = 0;

  int chunk_size = str_len / com.size();
  int start = com.rank() * chunk_size;
  int end = (com.rank() == com.size() - 1) ? str_len : start + chunk_size;

  for (int i = start; i < end; i++) {
    if (str1[i] != str2[i]) {
      local_count++;
    }
  }
  reduce(com, local_count, res, std::plus(), 0);

  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }
  return true;
}
