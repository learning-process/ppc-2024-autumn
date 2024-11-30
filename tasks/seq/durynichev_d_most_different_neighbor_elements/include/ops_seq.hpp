#pragma once

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

namespace durynichev_d_most_different_neighbor_elements_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input, result;
};

}  // namespace durynichev_d_most_different_neighbor_elements_seq