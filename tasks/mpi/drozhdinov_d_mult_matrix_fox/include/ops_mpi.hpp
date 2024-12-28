// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/archive/basic_archive.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace drozhdinov_d_mult_matrix_fox_mpi {
std::vector<double> SequentialFox(const std::vector<double>& A, const std::vector<double>& B, int k, int l, int n);
void SimpleMult(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int block);
std::vector<double> paddingMatrix(const std::vector<double>& mat, int rows, int cols, int padding);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int k{}, l{}, m{}, n{};
  std::vector<double> A, B, C;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::vector<double> ParallelFox(const std::vector<double>& A, const std::vector<double>& B, int k, int l, int n);

 private:
  int k{}, l{}, m{}, n{};
  std::vector<double> A, B, C;
  boost::mpi::communicator world;
};

}  // namespace drozhdinov_d_mult_matrix_fox_mpi