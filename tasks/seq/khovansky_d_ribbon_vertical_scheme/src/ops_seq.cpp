// Copyright 2024 Khovansky Dmitry
#include "seq/khovansky_d_ribbon_vertical_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>

using namespace std::chrono_literals;

bool khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq::validation() {
  internal_order_test();

  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
}

bool khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq::pre_processing() {
  internal_order_test();

  hello_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  hello_vector = reinterpret_cast<int*>(taskData->inputs[1]);

  int matrix_elements_count = taskData->inputs_count[0];
  rows_count = taskData->inputs_count[1];
  columns_count = matrix_elements_count / rows_count;

  goodbye_vector.assign(columns_count, 0);

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq::run() {
  internal_order_test();

  for (int i = 0; i < rows_count; ++i) {
    for (int j = 0; j < columns_count; ++j) {
      goodbye_vector[j] += hello_matrix[i * columns_count + j] * hello_vector[i];
    }
  }

  return true;
}

bool khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq::post_processing() {
  internal_order_test();

  int* result = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(goodbye_vector.begin(), goodbye_vector.end(), result);

  return true;
}
