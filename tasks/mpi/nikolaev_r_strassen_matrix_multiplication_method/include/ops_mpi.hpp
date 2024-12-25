#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_strassen_matrix_multiplication_method_mpi {

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

class StrassenMatrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit StrassenMatrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  std::vector<double> strassen_mpi(const std::vector<double>& A, const std::vector<double>& B, size_t n);

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;

  boost::mpi::communicator world;
};

std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> strassen_seq(const std::vector<double>& A, const std::vector<double>& B, size_t n);
}  // namespace nikolaev_r_strassen_matrix_multiplication_method_mpi
