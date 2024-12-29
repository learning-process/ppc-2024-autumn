#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace polikanov_v_gradient_method_mpi {

inline std::vector<double> MultiplyMatrixByVector(boost::mpi::communicator& world, std::vector<double>& flat_matrix,
                                                  std::vector<double>& vector, int vsize) {
  int rank = world.rank();
  int size = world.size();
  std::vector<int> rows_per_process(size, vsize / size);
  for (int i = 0; i < vsize % size; ++i) {
    rows_per_process[i]++;
  }
  std::vector<int> displs(size, 0);
  std::vector<int> sizes(size, 0);
  for (int i = 0; i < size; ++i) {
    sizes[i] = rows_per_process[i] * vsize;
    if (i > 0) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }
  int local_rows = rows_per_process[rank];
  std::vector<double> local_matrix(local_rows * vsize);
  scatterv(world, flat_matrix.data(), sizes, displs, local_matrix.data(), sizes[rank], 0);
  broadcast(world, vector, 0);
  std::vector<double> local_result(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < vsize; ++j) {
      local_result[i] += local_matrix[i * vsize + j] * vector[j];
    }
  }
  std::vector<double> result;
  if (rank == 0) {
    result.resize(vsize);
  }
  std::vector<int> recv_sizes = rows_per_process;
  std::vector<int> recv_displs(size, 0);
  for (int i = 1; i < size; ++i) {
    recv_displs[i] = recv_displs[i - 1] + recv_sizes[i - 1];
  }
  gatherv(world, local_result.data(), local_result.size(), result.data(), recv_sizes, recv_displs, 0);
  broadcast(world, result, 0);

  return result;
}

inline double VectorNorm(const std::vector<double>& vector) {
  double sum = 0.0;
  for (double value : vector) {
    sum += value * value;
  }
  return sqrt(sum);
}

inline double DotProduct(boost::mpi::communicator& world, const std::vector<double>& vector1,
                         const std::vector<double>& vector2) {
  int rank = world.rank();
  int size = world.size();
  size_t global_size = vector1.size();
  std::vector<int> sizes(size, global_size / size);
  std::vector<int> displs(size, 0);
  for (size_t i = 0; i < global_size % size; ++i) {
    sizes[i]++;
  }
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sizes[i - 1];
  }
  std::vector<double> local_v1(sizes[rank]);
  std::vector<double> local_v2(sizes[rank]);
  scatterv(world, vector1.data(), sizes, displs, local_v1.data(), sizes[rank], 0);
  scatterv(world, vector2.data(), sizes, displs, local_v2.data(), sizes[rank], 0);
  double local_sum = 0.0;
  for (size_t i = 0; i < local_v1.size(); ++i) {
    local_sum += local_v1[i] * local_v2[i];
  }
  double global_sum = 0.0;
  all_reduce(world, local_sum, global_sum, std::plus<>());

  return global_sum;
}

inline std::vector<double> ConjugateGradientMethod(boost::mpi::communicator& world, std::vector<double>& flat_matrix,
                                                   std::vector<double>& rhs, std::vector<double> solution,
                                                   double tolerance, int size) {
  std::vector<double> matrixTimesSolution = MultiplyMatrixByVector(world, flat_matrix, solution, size);
  std::vector<double> residual(size);
  std::vector<double> direction(size);
  for (int i = 0; i < size; ++i) {
    residual[i] = rhs[i] - matrixTimesSolution[i];
  }
  double residualNormSquared = DotProduct(world, residual, residual);
  if (sqrt(residualNormSquared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrixTimesDirection(size);
  while (sqrt(residualNormSquared) > tolerance) {
    matrixTimesDirection = MultiplyMatrixByVector(world, flat_matrix, direction, size);
    double directionDotMatrixTimesDirection = DotProduct(world, direction, matrixTimesDirection);
    double alpha = residualNormSquared / directionDotMatrixTimesDirection;
    for (int i = 0; i < size; ++i) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * matrixTimesDirection[i];
    }
    double newResidualNormSquared = DotProduct(world, residual, residual);
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
  boost::mpi::communicator world;
};

}  // namespace polikanov_v_gradient_method_mpi