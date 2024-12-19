// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

namespace drozhdinov_d_mult_matrix_fox_seq {
std::vector<double> SequentialFox(const std::vector<double>& A, const std::vector<double>& B, int k, int l, int n);
void SimpleMult(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int block);
std::vector<double> paddingMatrix(const std::vector<double>& mat, int rows, int cols, int padding);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int k{}, l{}, m{}, n{};
  std::vector<double> A, B, C;
};

}  // namespace drozhdinov_d_mult_matrix_fox_seq