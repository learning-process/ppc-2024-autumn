// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shlyakov_m_min_value_of_row_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<std::vector<int>> get_random_matr(int sz_row, int sz_col);

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

}  // namespace shlyakov_m_min_value_of_row_seq