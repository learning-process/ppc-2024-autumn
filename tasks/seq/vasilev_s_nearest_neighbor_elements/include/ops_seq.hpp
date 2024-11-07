#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_nearest_neighbor_elements_seq {

class FindClosestNeighborsSequential : public ppc::core::Task {
 public:
  explicit FindClosestNeighborsSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int min_diff_{};
  int index1_{};
  int index2_{};
};

}  // namespace vasilev_s_nearest_neighbor_elements_seq
