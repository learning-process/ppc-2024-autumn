#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace rams_s_gaussian_elimination_horizontally_seq {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  int rows_count;
  int cols_count;
  std::vector<double> res;
};

}  // namespace rams_s_gaussian_elimination_horizontally_seq
