// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_min_val_row_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> generate_rnd_vector(int size, int lower_bound = 0, int upper_bound = 50);
  static std::vector<std::vector<int>> generate_rnd_matrix(int rows, int cols);

 private:
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

}  // namespace korovin_n_min_val_row_matrix_seq