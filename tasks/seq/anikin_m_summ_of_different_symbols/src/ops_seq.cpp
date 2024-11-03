// Copyright 2024 Anikin Maksim
#include "seq/anikin_m_summ_of_different_symbols/include/ops_seq.hpp"

#include <vector>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential::pre_processing() {
  internal_order_test();
  // Input
  input.push_back(reinterpret_cast<char*>(taskData->inputs[0]));
  input.push_back(reinterpret_cast<char*>(taskData->inputs[1]));
  // output
  res = 0;
  return true;
}

bool anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential::run() {
  internal_order_test();
  auto i1 = input[0].begin();
  auto i2 = input[1].begin();
  while(i1 != input[0].end() || i2 != input[1].end()) {
    if(*i1 != *i2) {
      res++;
    }
    if(i1 != input[0].end()) i1++;
    if(i2 != input[1].end()) i2++;
  }
  return true;
}

bool anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
