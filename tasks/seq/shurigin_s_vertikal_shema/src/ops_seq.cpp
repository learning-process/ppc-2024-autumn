#include "seq/shurigin_s_vertikal_shema/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

namespace shurigin_s_vertikal_shema {

bool TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1;
}

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  matrix_data_ = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  num_cols_ = taskData->inputs_count[1];

  num_rows_ = matrix_size / num_cols_;

  input_vector_.assign(vector_data, vector_data + num_cols_);

  result_vector_.assign(num_rows_, 0);

  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();
  for (int j = 0; j < num_cols_; ++j) {
    for (int i = 0; i < num_rows_; ++i) {
      result_vector_[i] += matrix_data_[j * num_rows_ + i] * input_vector_[j];
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}

}  // namespace shurigin_s_vertikal_shema
