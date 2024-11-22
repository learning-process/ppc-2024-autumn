// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muhina_m_horizontal_cheme_mpi {
std::vector<int> matrixVectorMultiplication(const std::vector<int>& matrix, const std::vector<int>& vec, int rows,
                                            int cols);
void calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);

class HorizontalSchemeMPISequential : public ppc::core::Task {
 public:
  explicit HorizontalSchemeMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> vec_;
  std::vector<int> result_;
};

class HorizontalSchemeMPIParallel : public ppc::core::Task {
 public:
  explicit HorizontalSchemeMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_;
  std::vector<int> vec_;
  std::vector<int> result_;
  int rows_;
  int cols_;
  std::vector<int> distribution;
  std::vector<int> displacement;
  boost::mpi::communicator world_;
};

}  // namespace muhina_m_horizontal_cheme_mpi
