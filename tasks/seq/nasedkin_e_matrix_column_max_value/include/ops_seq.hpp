#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_matrix_column_max_value_seq {

class MatrixColumnMaxTaskSequential : public ppc::core::Task {
 public:
  explicit MatrixColumnMaxTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

}  // namespace nasedkin_e_matrix_column_max_value_seq
