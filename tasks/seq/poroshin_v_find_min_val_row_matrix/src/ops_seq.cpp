// Copyright 2024 Nesterov Alexander
#include "seq/poroshin_v_find_min_val_row_matrix/include/ops_seq.hpp"

#include <limits>  // for INT_MAX and INT_MIN
#include <thread>

using namespace std::chrono_literals;

bool poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int m = taskData->inputs_count[0];
  int n = taskData->inputs_count[1];
  int size = m * n;

  input_.resize(size);
  res.resize(m);

  for (int i = 0; i < size; i++) {
    input_[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  int m = taskData->inputs_count[0];
  int n = taskData->inputs_count[1];

  int mn;
  for (int i = 0; i < m; i++) {
    mn = std::numeric_limits<int>::max();  // Use std::numeric_limits for INT_MAX
    for (int j = n * i; j < n * i + n; j++) {
      mn = std::min(mn, input_[j]);
    }
    res[i] = mn;
  }

  return true;
}

bool poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

std::vector<int> poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential::gen(int m, int n) {
  std::vector<int> tmp(m * n);
  int n1 = std::max(n, m);
  int m1 = std::min(n, m);

  for (auto& t : tmp) {
    t = n1 + (std::rand() % (m1 - n1 + 7));
  }

  for (int i = 0; i < m; i++) {
    tmp[(std::rand() % n) + i * n] =
        std::numeric_limits<int>::min();  // In 1 of n columns, the value must be INT_MIN (needed to check answer)
  }

  return tmp;
}