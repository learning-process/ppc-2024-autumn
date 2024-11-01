// Copyright 2023 Nesterov Alexander
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

namespace zinoviev_a_sum_cols_matrix_mpi {

std::vector<int> generateRandomVector(int size);
int computeLinearCoordinates(int xCoord, int yCoord, int width);
std::vector<int> calculateMatrixSumSequential(const std::vector<int>& matrix, int width, int height, int startX,
                                              int endX);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> data) : Task(std::move(data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  int numCols{};
  int numRows{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> data) : Task(std::move(data)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  int numCols{};
  int numRows{};
  boost::mpi::communicator mpiWorld;
};

}  // namespace zinoviev_a_sum_cols_matrix_mpi
