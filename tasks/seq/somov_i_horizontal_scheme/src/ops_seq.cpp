#include "seq/somov_i_horizontal_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace somov_i_horizontal_scheme {

MatrixVectorTask::MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

bool MatrixVectorTask::pre_processing() {
  internal_order_test();

  uint32_t rowCount = taskData->inputs_count[0];
  uint32_t colCount = taskData->inputs_count[1];

  matrix_.resize(rowCount, std::vector<int32_t>(colCount));
  vector_.resize(colCount);
  result_.resize(rowCount);

  int32_t* data;
  for (uint32_t i = 0; i < rowCount; ++i) {
    data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    std::copy(data, data + colCount, matrix_[i].begin());
  }

  data = reinterpret_cast<int32_t*>(taskData->inputs[rowCount]);
  std::copy(data, data + colCount, vector_.begin());

  return true;
}

bool MatrixVectorTask::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool MatrixVectorTask::run() {
  internal_order_test();

  for (uint32_t i = 0; i < matrix_.size(); ++i) {
    result_[i] = 0;
    for (uint32_t j = 0; j < vector_.size(); ++j) {
      result_[i] += matrix_[i][j] * vector_[j];
    }
  }

  return true;
}

bool MatrixVectorTask::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));

  return true;
}

}  // namespace somov_i_horizontal_scheme