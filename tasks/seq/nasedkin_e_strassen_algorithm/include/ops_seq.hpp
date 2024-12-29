#pragma once

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm {

class StrassenAlgorithmSEQ : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  static std::vector<double> strassen_multiply_seq(const std::vector<double>& matrixA,
                                                   const std::vector<double>& matrixB, size_t size);

  std::vector<double> inputMatrixA;
  std::vector<double> inputMatrixB;
  std::vector<double> outputMatrix;
  size_t matrixSize;
};

std::vector<double> strassen_recursive(const std::vector<double>& matrixA, const std::vector<double>& matrixB,
                                       size_t size);
std::vector<double> matrix_add(const std::vector<double>& matrixA, const std::vector<double>& matrixB, size_t size);
std::vector<double> matrix_subtract(const std::vector<double>& matrixA, const std::vector<double>& matrixB,
                                    size_t size);

size_t nextPowerOfTwo(size_t n);
std::vector<double> padMatrix(const std::vector<double>& matrix, size_t originalSize, size_t newSize);
}  // namespace nasedkin_e_strassen_algorithm