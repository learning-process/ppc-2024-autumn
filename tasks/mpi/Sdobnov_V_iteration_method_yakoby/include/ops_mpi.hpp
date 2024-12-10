// Copyright 2024 Sdobnov Vladimir
#pragma once
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Sdobnov_iteration_method_yakoby_par {

std::vector<double> iteration_method_yakoby(int n, const std::vector<double>& A, const std::vector<double>& b,
                                            double tolerance = 1e-6, int maxIterations = 1000);

class IterationMethodYakobySeq : public ppc::core::Task {
 public:
  explicit IterationMethodYakobySeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix_;
  std::vector<double> free_members_;
  std::vector<double> res_;
  int size_;
  double tolerance = 1e-6;
  int maxIterations = 100;
};

class IterationMethodYakobyPar : public ppc::core::Task {
 public:
  explicit IterationMethodYakobyPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix_;
  std::vector<double> free_members_;
  std::vector<double> res_;

  std::vector<int> mat_part_sizes;
  std::vector<int> mat_part_offsets;
  std::vector<int> free_members_part_sizes;
  std::vector<int> free_members_part_offsets;

  std::vector<double> l_matrix;
  std::vector<double> l_free_members;
  std::vector<double> l_res;

  std::vector<double> last_res;

  int size_;
  double tolerance = 1e-6;
  int maxIterations = 100;
  boost::mpi::communicator world;
};
}  // namespace Sdobnov_iteration_method_yakoby_par