#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace somov_i_horizontal_scheme {

class MatrixVectorTask : public ppc::core::Task {
 public:
  explicit MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData);

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  uint32_t getRowCount() const { return rowCount_; }
  void setRowCount(uint32_t rowCount) { rowCount_ = rowCount; }

  uint32_t getColCount() const { return colCount_; }
  void setColCount(uint32_t colCount) { colCount_ = colCount; }

 private:
  std::vector<int32_t> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;
  uint32_t rowCount_;
  uint32_t colCount_;
};

}  // namespace somov_i_horizontal_scheme
