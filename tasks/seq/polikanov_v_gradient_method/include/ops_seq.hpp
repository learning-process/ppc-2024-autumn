#pragma once

#include <cmath>

#include "core/task/include/task.hpp"

namespace polikanov_v_gradient_method_seq {

inline std::vector<double> MultiplyMatrixByVector(const std::vector<double>& flat_matrix,
                                                  const std::vector<double>& vector, int size) {
  std::vector<double> result(size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      result[i] += flat_matrix[i * size + j] * vector[j];
    }
  }
  return result;
}

inline double VectorNorm(const std::vector<double>& vector) {
  double sum = 0.0;
  for (double value : vector) {
    sum += value * value;
  }
  return sqrt(sum);
}

inline double DotProduct(const std::vector<double>& vector1, const std::vector<double>& vector2) {
  double sum = 0.0;
  for (size_t i = 0; i < vector1.size(); ++i) {
    sum += vector1[i] * vector2[i];
  }
  return sum;
}

inline std::vector<double> ConjugateGradientMethod(std::vector<double>& flat_matrix, std::vector<double>& rhs,
                                                   std::vector<double> solution, double tolerance, int size) {
  std::vector<double> matrixTimesSolution = MultiplyMatrixByVector(flat_matrix, solution, size);
  std::vector<double> residual(size);
  std::vector<double> direction(size);

  for (int i = 0; i < size; ++i) {
    residual[i] = rhs[i] - matrixTimesSolution[i];
  }

  double residualNormSquared = DotProduct(residual, residual);
  if (sqrt(residualNormSquared) < tolerance) {
    return solution;
  }

  direction = residual;
  std::vector<double> matrixTimesDirection(size);

  while (sqrt(residualNormSquared) > tolerance) {
    matrixTimesDirection = MultiplyMatrixByVector(flat_matrix, direction, size);
    double directionDotMatrixTimesDirection = DotProduct(direction, matrixTimesDirection);
    double alpha = residualNormSquared / directionDotMatrixTimesDirection;

    for (int i = 0; i < size; ++i) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * matrixTimesDirection[i];
    }

    double newResidualNormSquared = DotProduct(residual, residual);
    double beta = newResidualNormSquared / residualNormSquared;
    residualNormSquared = newResidualNormSquared;

    for (int i = 0; i < size; ++i) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }

  return solution;
}

class GradientMethod : public ppc::core::Task {
 public:
  explicit GradientMethod(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  double tolerance;
  std::vector<double> flat_matrix, rhs, initial_guess, result;
};

}  // namespace polikanov_v_gradient_method_seq