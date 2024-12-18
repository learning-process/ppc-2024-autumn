#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_bellman_ford_seq {

class BellmanFordSeq : public ppc::core::Task {
 public:
  explicit BellmanFordSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t V{};
  size_t E{};
  std::vector<int> values;
  std::vector<int> columns;
  std::vector<int> row_ptr;
  std::vector<int> shortest_paths;
  const int INF = std::numeric_limits<int>::max();

  bool Iteration(std::vector<int>& paths);
  bool check_negative_cycle();
  void toCRS(const int* input_matrix);
};

}  // namespace zinoviev_a_bellman_ford_seq