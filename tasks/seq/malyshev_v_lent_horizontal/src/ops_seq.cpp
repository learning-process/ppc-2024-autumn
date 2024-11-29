#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

bool malyshev_lent_horizontal::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];

  matrix_.resize(rows, std::vector<int32_t>(cols));
  vector_.resize(cols);
  result_.resize(rows);

  int32_t* data;
  for (uint32_t i = 0; i < matrix_.size(); i++) {
    data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    std::copy(data, data + cols, matrix_[i].data());
  }

  data = reinterpret_cast<int32_t*>(taskData->inputs[rows]);
  std::copy(data, data + cols, vector_.data());

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_lent_horizontal::TestTaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < matrix_.size(); i++) {
    result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); j++) {
      result_[i] += matrix_[i][j] * vector_[j];
    }
  }

  return true;
}

bool malyshev_lent_horizontal::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));

  return true;
}