// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace chizhov_m_dijkstra_seq {

void convertToCRS(const std::vector<std::vector<int>>& w, std::vector<int>& values, std::vector<int>& colIndex,
                  std::vector<int>& rowPtr);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
  int st{};
  int size{};
};

}  // namespace chizhov_m_dijkstra_seq