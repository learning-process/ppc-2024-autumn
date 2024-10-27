// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace grudzin_k_nearest_neighbor_elements_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void gen_random_vector(const std::vector<int>& v);

 private:
  std::vector<std::pair<int, int>> input_{};
  std::pair<int, int> res{};
};

}  // namespace grudzin_k_nearest_neighbor_elements_seq