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

namespace kolokolova_d_gaussian_method_horizontal_mpi {

std::vector<int> getRandomVector(int sz);
int find_rank(std::vector<double>& matrix, int rows, int cols);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coeff;
  std::vector<int> input_y;
  std::vector<double> res;
  int count_equations = 0;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coeff;
  std::vector<int> input_y;
  std::vector<double> res;

  std::vector<double> local_matrix;
  std::vector<double> changed_matrix;
  std::vector<double> matrix_argum;
  std::vector<double> local_max_row;
  std::vector<double> res_matrix;
  int count_equations = 0;
  int size_row = 0;
  int count_row_proc = 0;
  boost::mpi::communicator world;
};

}  // namespace kolokolova_d_gaussian_method_horizontal_mpi