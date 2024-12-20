#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

class StrassenAlgorithmSequential : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;
  std::vector<std::vector<double>> C_;
  size_t n;

  void multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  void strassen(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  static void add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  static void subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
};

class StrassenAlgorithmParallel : public ppc::core::Task {
 public:
  explicit StrassenAlgorithmParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;
  std::vector<std::vector<double>> C_;
  size_t n;

  std::vector<std::vector<double>> local_A;
  std::vector<std::vector<double>> local_B;
  std::vector<std::vector<double>> local_C;

  std::vector<int> sizes_a;
  std::vector<int> displs_a;
  std::vector<int> sizes_b;
  std::vector<int> displs_b;

  boost::mpi::communicator world;
  static void calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
  void distribute_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& local_matrix, const std::vector<int>& sizes, const std::vector<int>& displs);
  void gather_matrix(std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& local_matrix, const std::vector<int>& sizes, const std::vector<int>& displs);
  void strassen_mpi(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  static void add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  static void subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C);
  static std::vector<double> flatten_matrix(const std::vector<std::vector<double>>& matrix);
  static std::vector<std::vector<double>> unflatten_matrix(const std::vector<double>& flat, size_t rows, size_t cols);
};

}  // namespace nasedkin_e_strassen_algorithm_mpi