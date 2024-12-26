#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace savchenko_m_matrix_mult_strassen_mpi {
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_C;
  size_t dims;

  static bool is_power_of_two(size_t _size);
  static std::vector<double> add_matrices(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
  static std::vector<double> sub_matrices(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
  static std::vector<double> multiply_standard(const std::vector<double>& A, const std::vector<double>& B,
                                               size_t _size);
  std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_C;
  size_t dims;

  static bool is_power_of_two(size_t _size);
  static std::vector<double> add_matrices(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
  static std::vector<double> sub_matrices(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
  static std::vector<double> multiply_standard(const std::vector<double>& A, const std::vector<double>& B,
                                               size_t _size);
  std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
  std::vector<double> strassen_parallel(const std::vector<double>& A, const std::vector<double>& B, size_t _size);
};

}  // namespace savchenko_m_matrix_mult_strassen_mpi