#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_min_value_by_columns_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rs;
  int cs;
  std::vector<std::vector<int>> input_;
  std::vector<int> res;
};

}  // namespace dormidontov_e_min_value_by_columns_seq