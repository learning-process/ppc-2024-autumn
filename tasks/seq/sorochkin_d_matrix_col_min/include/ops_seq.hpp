#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace sorochkin_d_matrix_col_min_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), rows_(0), cols_(0) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t rows_, cols_;
  std::vector<int> input_, res_;
};

}  // namespace sorochkin_d_matrix_col_min_seq