#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_jacobi_method_mpi {

class JacobiMethodBaseTask : public ppc::core::Task {
 public:
  explicit JacobiMethodBaseTask(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  virtual ~JacobiMethodBaseTask() = default;

  bool pre_processing() override = 0;
  bool validation() override = 0;
  bool run() override = 0;
  bool post_processing() override = 0;

 protected:
  std::vector<double> matrix_A;
  std::vector<double> vector_b;
  std::vector<double> solution_x;
  size_t matrix_size;

  size_t maxIterations = 2000;
  double epsilon = 1e-5;

  bool has_converged(const std::vector<double>& x_old, const std::vector<double>& x_new) const {
    double norm_diff_squared = 0.0;
    double norm_new_squared = 0.0;

    for (size_t i = 0; i < x_old.size(); ++i) {
      double diff = x_new[i] - x_old[i];
      norm_diff_squared += diff * diff;
      norm_new_squared += x_new[i] * x_new[i];
    }

    if (norm_new_squared == 0.0) {
      return false;
    }

    double relative_error = sqrt(norm_diff_squared / norm_new_squared);
    return relative_error < epsilon;
  }
};

class JacobiMethodSequentialTask : public JacobiMethodBaseTask {
 public:
  explicit JacobiMethodSequentialTask(std::shared_ptr<ppc::core::TaskData> taskData)
      : JacobiMethodBaseTask(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

class JacobiMethodParallelTask : public JacobiMethodBaseTask {
 public:
  explicit JacobiMethodParallelTask(std::shared_ptr<ppc::core::TaskData> taskData)
      : JacobiMethodBaseTask(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> x_prev;
  std::vector<double> local_A;
  std::vector<double> local_b;

  std::vector<int> sizes_a;
  std::vector<int> displs_a;
  std::vector<int> sizes_b;
  std::vector<int> displs_b;

  boost::mpi::communicator world;

  static void calculate_distribution(int total_size, int num_procs, std::vector<int>& sizes, std::vector<int>& displs,
                                     int block_size);
  static bool is_singular(const std::vector<double>& A, size_t matrix_size);
};

}  // namespace kavtorev_d_jacobi_method_mpi