#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_conjugate_gradient_method {

inline std::vector<double> MatrixVectorProduct(boost::mpi::communicator& world, std::vector<double>& matrix,
                                               std::vector<double>& vector, int size) {
  int rank = world.rank();
  int num_procs = world.size();
  std::vector<int> rows_per_proc(num_procs, size / num_procs);
  for (int i = 0; i < size % num_procs; ++i) {
    rows_per_proc[i]++;
  }
  std::vector<int> displs(num_procs, 0);
  std::vector<int> sizes(num_procs, 0);
  for (int i = 0; i < num_procs; ++i) {
    sizes[i] = rows_per_proc[i] * size;
    if (i > 0) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }
  int local_rows = rows_per_proc[rank];
  std::vector<double> local_matrix(local_rows * size);
  scatterv(world, matrix.data(), sizes, displs, local_matrix.data(), sizes[rank], 0);
  broadcast(world, vector, 0);
  std::vector<double> local_result(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < size; ++j) {
      local_result[i] += local_matrix[i * size + j] * vector[j];
    }
  }
  std::vector<double> result;
  if (rank == 0) {
    result.resize(size);
  }
  std::vector<int> recv_sizes = rows_per_proc;
  std::vector<int> recv_displs(num_procs, 0);
  for (int i = 1; i < num_procs; ++i) {
    recv_displs[i] = recv_displs[i - 1] + recv_sizes[i - 1];
  }
  gatherv(world, local_result.data(), local_result.size(), result.data(), recv_sizes, recv_displs, 0);
  broadcast(world, result, 0);

  return result;
}

inline double ComputeVectorNorm(const std::vector<double>& vector) {
  double sum = 0.0;
  for (double value : vector) {
    sum += value * value;
  }
  return sqrt(sum);
}

inline double DotProduct(boost::mpi::communicator& world, const std::vector<double>& vector1,
                         const std::vector<double>& vector2) {
  int rank = world.rank();
  int num_procs = world.size();
  size_t global_size = vector1.size();
  std::vector<int> sizes(num_procs, global_size / num_procs);
  std::vector<int> displs(num_procs, 0);
  for (size_t i = 0; i < global_size % num_procs; ++i) {
    sizes[i]++;
  }
  for (int i = 1; i < num_procs; ++i) {
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

inline std::vector<double> SolveConjugateGradient(boost::mpi::communicator& world, std::vector<double>& matrix,
                                                  std::vector<double>& rhs, std::vector<double> solution,
                                                  double tolerance, int size) {
  std::vector<double> matrix_times_solution = MatrixVectorProduct(world, matrix, solution, size);
  std::vector<double> residual(size);
  std::vector<double> direction(size);
  for (int i = 0; i < size; ++i) {
    residual[i] = rhs[i] - matrix_times_solution[i];
  }
  double residual_norm_squared = DotProduct(world, residual, residual);
  if (sqrt(residual_norm_squared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrix_times_direction(size);
  while (sqrt(residual_norm_squared) > tolerance) {
    matrix_times_direction = MatrixVectorProduct(world, matrix, direction, size);
    double direction_dot_matrix_times_direction = DotProduct(world, direction, matrix_times_direction);
    double alpha = residual_norm_squared / direction_dot_matrix_times_direction;
    for (int i = 0; i < size; ++i) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * matrix_times_direction[i];
    }
    double new_residual_norm_squared = DotProduct(world, residual, residual);
    double beta = new_residual_norm_squared / residual_norm_squared;
    residual_norm_squared = new_residual_norm_squared;
    for (int i = 0; i < size; ++i) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }

  return solution;
}

class ConjugateGradientSolver : public ppc::core::Task {
 public:
  explicit ConjugateGradientSolver(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  double tolerance;
  std::vector<double> matrix, rhs, initial_guess, result;
  boost::mpi::communicator world;
};

}  // namespace malyshev_v_conjugate_gradient_method