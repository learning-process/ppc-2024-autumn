#pragma once

#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_strassen_matrix_multiplication_method_seq {

class StrassenMatrixMultiplicationSequential : public ppc::core::Task {
 public:
  explicit StrassenMatrixMultiplicationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;
};

std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, size_t n);
}  // namespace nikolaev_r_strassen_matrix_multiplication_method_seq
