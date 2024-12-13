// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_matrix_multiplication_seq {

std::vector<int> Multiplication(size_t M, size_t N, size_t K, const std::vector<int>& A, const std::vector<int>& B);

class matrixMultiplication : public ppc::core::Task {
 public:
  explicit matrixMultiplication(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrixA;
  std::vector<int> matrixB;
  std::vector<int> matrixC;

  size_t lineA{};
  size_t columnA{};

  size_t lineB{};
  size_t columnB{};
};
}  // namespace frolova_e_matrix_multiplication_seq