// Copyright 2024 Nesterov Alexander
// shlyakov_m_min_value_of_row
#include "seq/shlyakov_m_min_value_of_row/include/ops_seq.hpp"

#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool shlyakov_m_min_value_of_row_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  size_t sz_row = taskData->inputs_count[0];
  size_t sz_col = taskData->inputs_count[1];
  input_.resize(sz_row, std::vector<int>(sz_col));

  for (size_t i = 0; i < sz_row; i++) {
    auto* matr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (size_t j = 0; j < sz_col; j++) {
      input_[i][j] = matr[j];
    }
  }
  res_.resize(sz_row);

  return true;
}

bool shlyakov_m_min_value_of_row_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
       (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0)) &&
      (taskData->outputs_count[0] == taskData->inputs_count[0]))
    return (true);

  return (false);
}

bool shlyakov_m_min_value_of_row_seq::TestTaskSequential::run() {
  internal_order_test();
  int min;
  size_t sz_row = input_.size();
  size_t sz_col = input_[0].size();

  for (size_t i = 0; i < sz_row; i++) {
    min = input_[i][0];
    for (size_t j = 1; j < sz_col; j++) {
      if (input_[i][j] < min) {
        min = input_[i][j];
      }
    }
    res_[i] = min;
  }

  return true;
}

bool shlyakov_m_min_value_of_row_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  int* result = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    result[i] = res_[i];
  }

  return true;
}

std::vector<std::vector<int>> shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(int sz_row,
                                                                                                   int sz_col) {
  std::vector<int> rand_vec(sz_row);
  std::vector<std::vector<int>> rand_matr(sz_row, std::vector<int>(sz_col));

  for (auto& row : rand_matr) {
    for (auto& el : rand_vec) el = std::rand() % (1001) - 500;
    row = rand_vec;
    row[std::rand() % sz_col] = INT_MIN;
  }

  return rand_matr;
}