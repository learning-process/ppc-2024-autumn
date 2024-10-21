// Copyright 2023 Nesterov Alexander
#pragma once
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace Shurygin_S_max_po_stolbam_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> generating_random_vector(int size, int lower_bound = 0, int upper_bound = 10);
  static std::vector<std::vector<int>> generate_random_matrix(int rows, int columns);

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

}  // namespace Shurygin_S_max_po_stolbam_matrix_seq
