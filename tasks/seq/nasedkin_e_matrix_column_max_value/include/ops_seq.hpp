#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "core/task/include/task.hpp"

namespace nasedkin_e_matrix_column_max_value_seq {

std::vector<int> getRandomMatrix(int rows, int cols);

class MatrixColumnMaxSeq : public ppc::core::Task {
 public:
  explicit MatrixColumnMaxSeq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};

}  // namespace nasedkin_e_matrix_column_max_value_seq