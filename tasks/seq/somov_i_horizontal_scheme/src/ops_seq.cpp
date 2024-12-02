#include "seq/somov_i_horizontal_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace somov_i_horizontal_scheme {

MatrixVectorTask::MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData)
    : Task(std::move(taskData)), rowCount_(0), colCount_(0) {}

bool MatrixVectorTask::pre_processing() {
  internal_order_test();

  matrix_.resize(rowCount_ * colCount_);
  vector_.resize(colCount_);
  result_.resize(rowCount_);

  int32_t* matrixData = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  int32_t* vectorData = reinterpret_cast<int32_t*>(taskData->inputs[1]);

  matrix_.assign(matrixData, matrixData + rowCount_ * colCount_);
  vector_.assign(vectorData, vectorData + colCount_);

  return true;
}

bool MatrixVectorTask::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == rowCount_ && taskData->inputs_count.size() == 2;
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
