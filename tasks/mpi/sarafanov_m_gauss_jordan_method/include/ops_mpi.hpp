#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sarafanov_m_gauss_jordan_method_mpi {

std::vector<double> processMatrix(int n, int k, const std::vector<double>& matrix);
void calcSizesDispls(int n, int k, int world_size, std::vector<int>& sizes, std::vector<int>& displs);
std::vector<std::pair<int, int>> getIndicies(int rows, int cols);
void updateMatrix(int n, int k, std::vector<double>& matrix, const std::vector<double>& iter_result);

class GaussJordanMethodParallelMPI : public ppc::core::Task {
 public:
  explicit GaussJordanMethodParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
  std::vector<int> sizes;
  std::vector<int> displs;
  std::vector<double> iter_matrix;
  std::vector<double> iter_result;
  std::vector<std::pair<int, int>> indicies;
  boost::mpi::communicator world;
};

class GaussJordanMethodSequentialMPI : public ppc::core::Task {
 public:
  explicit GaussJordanMethodSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
  std::vector<double> iter_matrix;
  std::vector<double> iter_result;
};

}  // namespace sarafanov_m_gauss_jordan_method_mpi
