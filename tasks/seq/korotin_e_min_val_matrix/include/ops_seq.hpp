// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_min_val_matrix_seq {

std::vector<double> getRandomMatrix(unsigned rows, unsigned columns, double scal);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_;
  double res{};
};

}  // namespace korotin_e_min_val_matrix_seq
