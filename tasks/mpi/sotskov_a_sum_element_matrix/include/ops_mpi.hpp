#pragma once
#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "core/task/include/task.hpp"

namespace sotskov_a_sum_element_matrix_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_matrix(const std::vector<double>& matrix, int rows, int cols);

 private:
  double sum_elements(const std::vector<double>& matrix);

  std::vector<double> matrix_;
  int rows_{};
  int cols_{};
  double result_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_matrix(const std::vector<double>& matrix, int rows, int cols);

 private:
  double parallel_sum_elements(const std::vector<double>& matrix);

  std::vector<double> matrix_;
  int rows_{};
  int cols_{};
  double local_result_{};
  double global_result_{};

  boost::mpi::communicator world;
};

}  // namespace sotskov_a_sum_element_matrix_mpi
