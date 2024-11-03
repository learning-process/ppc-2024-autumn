// Copyright 2024 Anikin Maksim
#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::string anikin_m_summ_of_different_symbols_mpi::getRandomString(int sz) {
  std::string alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM";
  std::string str = "";
  for (int i = 0; i < sz; i++) {
    str += alphabet[std::rand() % alphabet.size()];
  }
  return str;
}

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
    input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
  }
  // Init value for output
  res = 0;
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
  size_t loc_size = 0;
  if (com.rank() == 0) {
    loc_size = ((strlen(input[0])) + com.size() - 1) / com.size();
  }
  broadcast(com, loc_size, 0);
  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      int send_size = (loc_size <= strlen(input[0] - pr * loc_size)) ? loc_size : strlen(input[0] - pr * loc_size);
      com.send(pr, 0, input[0] + pr * loc_size, send_size);
      com.send(pr, 0, input[1] + pr * loc_size, send_size);
    }
  }
  if (com.rank() == 0) {
    std::string str1(input[0], loc_size);
    local_input.push_back(str1);
    std::string str2(input[1], loc_size);
    local_input.push_back(str2);
  } else {
    std::string str1('0', loc_size);
    com.recv(0, 0, str1.data(), loc_size);
    local_input.push_back(str1);
    std::string str2('0', loc_size);
    com.recv(0, 0, str2.data(), loc_size);
    local_input.push_back(str2);
  }
  int size_1 = local_input[0].size();
  int loc_res = 0;
  for (int i = 0; i < size_1; i++) {
    if (local_input[0][i] != local_input[1][i]) {
      loc_res += 1;
    }
  }
  reduce(com, loc_res, res, std::plus(), 0);
  return true;
}

bool anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }
  return true;
}
