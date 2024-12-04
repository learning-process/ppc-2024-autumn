#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_bellman_ford_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct Edge {
    int src, dest, weight;
  };

  std::vector<Edge> edges_;
  std::vector<int> distances_;
  int vertices_{0}, edges_count_{0}, source_{0};

  static constexpr int INF = std::numeric_limits<int>::max();
};

}  // namespace vavilov_v_bellman_ford_seq
