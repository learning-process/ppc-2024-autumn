#include "seq/kolokolova_d_max_of_row_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool kolokolova_d_max_of_row_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  size_t row_count = static_cast<size_t>(*taskData->inputs[1]);
  size_t col_count = taskData->inputs_count[0] / row_count;

  input_.resize(row_count, std::vector<int>(col_count));

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < row_count; ++i) {
    for (size_t j = 0; j < col_count; ++j) {
      input_[i][j] = input_ptr[i * col_count + j];
    }
  }
  res.resize(row_count);
  return true;
}

bool kolokolova_d_max_of_row_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool kolokolova_d_max_of_row_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); ++i) {
    int max_value = input_[i][0];
    for (size_t j = 1; j < input_[i].size(); ++j) {
      if (input_[i][j] > max_value) {
        max_value = input_[i][j];
      }
    }
    res[i] = max_value;
  }
  return true;
}

bool kolokolova_d_max_of_row_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); ++i) {
    output_ptr[i] = res[i];
    std::cout << "Output Max[" << i << "]: " << res[i] << std::endl;  // Для отладки
  }
  return true;
}
