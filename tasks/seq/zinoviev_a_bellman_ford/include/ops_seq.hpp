#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_bellman_ford_seq {

class BellmanFordSeqTaskSequential : public ppc::core::Task {
 public:
  explicit BellmanFordSeqTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> row_pointers_;
  std::vector<int> col_indices_;
  std::vector<int> values_;
  std::vector<int> distances_;
};

}  // namespace zinoviev_a_bellman_ford_seq