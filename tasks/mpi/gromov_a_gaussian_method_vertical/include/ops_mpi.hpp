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

namespace gromov_a_gaussian_method_vertical_mpi {

std::vector<int> getRandomVector(int sz);
int matrix_rank(std::vector<double>& matrix, int rows, int columns, int band_width);

class MPIGaussVerticalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussVerticalSequential(std::shared_ptr<ppc::core::TaskData> taskData_, int band_width_)
      : Task(std::move(taskData_)), band_width(band_width_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coefficient;
  std::vector<int> input_rhs;
  std::vector<double> res;
  int equations = 0;
  int band_width;
};

class MPIGaussVerticalParallel : public ppc::core::Task {
 public:
  explicit MPIGaussVerticalParallel(std::shared_ptr<ppc::core::TaskData> taskData_, int band_width_)
      : Task(std::move(taskData_)), band_width(band_width_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coefficient;
  std::vector<int> input_rhs;
  std::vector<double> res;

  std::vector<double> local_matrix;
  std::vector<double> changed_matrix;
  std::vector<double> matrix_argument;
  std::vector<double> local_max_row;
  std::vector<double> res_matrix;

  int remainder = 0;
  int equations = 0;
  int size_row = 0;
  int count_row_proc = 0;
  int band_width;
  boost::mpi::communicator world;
};

}  // namespace gromov_a_gaussian_method_vertical_mpi
