#include "seq/somov_i_horizontal_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace somov_i_horizontal_scheme {

MatrixVectorTask::MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)), rowCount_(0), colCount_(0) {}

bool MatrixVectorTask::pre_processing() {
  assert(taskData->inputs[0] != nullptr);
  assert(taskData->inputs[1] != nullptr);

  internal_order_test();

  matrix_.resize(rowCount_ * colCount_);
  vector_.resize(colCount_);
  result_.resize(rowCount_);

  if (taskData == nullptr || taskData->inputs.size() < 2) {
    return false;
  }

  auto* matrixData = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  auto* vectorData = reinterpret_cast<int32_t*>(taskData->inputs[1]);

  if (matrixData == nullptr || vectorData == nullptr) {
    return false;
  }

  matrix_.assign(matrixData, matrixData + rowCount_ * colCount_);
  vector_.assign(vectorData, vectorData + colCount_);

  return true;
}

bool MatrixVectorTask::validation() {
  internal_order_test();

  if (taskData->inputs.size() < 2) return false;
  if (taskData->inputs_count[0] != rowCount_ || taskData->inputs_count[1] != colCount_) return false;
  return taskData->outputs_count[0] == rowCount_;
}

bool MatrixVectorTask::run() {
  internal_order_test();

  for (uint32_t i = 0; i < rowCount_; ++i) {
    result_[i] = 0;
    for (uint32_t j = 0; j < colCount_; ++j) {
      result_[i] += matrix_[i * colCount_ + j] * vector_[j];
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
