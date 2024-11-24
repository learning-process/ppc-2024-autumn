// Copyright 2024 Nesterov Alexander
#include "seq/muhina_m_horizontal_cheme/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

std::vector<int> muhina_m_horizontal_cheme_seq::matrixVectorMultiplication(const std::vector<int>& matrix,
                                                                           const std::vector<int>& vec, int rows,
                                                                           int cols) {
  std::vector<int> result(rows, 0);

  for (int i = 0; i < rows; ++i) {
    int row_result = 0;
    for (int j = 0; j < cols; ++j) {
      row_result += matrix[i * cols + j] * vec[j];
    }
    result[i] = row_result;
  }
  return result;
}

bool muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential::pre_processing() {
  internal_order_test();

  int* m_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int m_size = taskData->inputs_count[0];

  int* v_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int v_size = taskData->inputs_count[1];

  matrix_.assign(m_data, m_data + m_size);
  vec_.assign(v_data, v_data + v_size);

  return true;
}

bool muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential::validation() {
  internal_order_test();
  return (!(taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0));
}

bool muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential::run() {
  internal_order_test();
  int cols = taskData->inputs_count[1];
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  result_ = matrixVectorMultiplication(matrix_, vec_, rows, cols);
  return true;
}

bool muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential::post_processing() {
  internal_order_test();
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_.begin(), result_.end(), output_data);
  return true;
}
