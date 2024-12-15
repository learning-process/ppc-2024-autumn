#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_gauss_jordan_method_mpi {

std::vector<double> processMatrix(int n, int k, const std::vector<double>& matrix);
void updateMatrix(int n, int k, std::vector<double>& matrix, const std::vector<double>& iter_result);

class GaussJordanParallel : public ppc::core::Task {
 public:
  explicit GaussJordanParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
  std::vector<int> sizes;
  std::vector<int> displaces;
  std::vector<double> iteration_matrix;
  std::vector<double> result;
  std::vector<std::pair<int, int>> current_index;
  boost::mpi::communicator world;
};

}  // namespace vasenkov_a_gauss_jordan_method_mpi
