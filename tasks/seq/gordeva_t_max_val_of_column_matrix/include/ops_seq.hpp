// Copyright 2023 Nesterov Alexander
#pragma once

// #include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace gordeva_t_max_val_of_column_matrix_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::vector<int> gen_rand_vec(int size, int lower_bound = 0, int upper_bound = 30);
  static std::vector<std::vector<int>> gen_rand_matr(int rows, int cols);

 private:
  // int input_{}, res{};
  std::vector<std::vector<int>> input_;
  std::vector<int> res_;
};

}  // namespace gordeva_t_max_val_of_column_matrix_seq
