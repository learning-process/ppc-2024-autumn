#include "seq/vavilov_v_min_elements_in_columns_of_matrix/include/ops_seq.hpp"

#include <random>

using namespace std::chrono_literals;

bool vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  input_.resize(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; i++) {
    int* input_row = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) {
      input_[i][j] = input_row[j];
    }
  }
  res_.resize(cols);
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[1]));
}

bool vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  res_.resize(input_[0].size());

  for (size_t i = 0; i < input_[0].size(); i++) {
    int min = input_[0][i];
    for (size_t j = 1; j < input_.size(); j++) {
      if (input_[j][i] < min) {
        min = input_[j][i];
      }
    }
    res_[i] = min;
  }
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output_matr[i] = res_[i];
  }
  return true;
}
