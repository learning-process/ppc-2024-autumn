#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_simple_iteration_method_mpi {
bool is_singular(const std::vector<double>& A, size_t n);
bool is_diagonally_dominant(const std::vector<double>& A, size_t n);

class SimpleIterationMethodSequential : public ppc::core::Task {
 public:
  explicit SimpleIterationMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  double tolerance_ = 1e-6;
  size_t max_iterations_ = 1500;
};

class SimpleIterationMethodParallel : public ppc::core::Task {
 public:
  explicit SimpleIterationMethodParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  double tolerance_ = 1e-6;
  size_t max_iterations_ = 1500;

  boost::mpi::communicator world;
};
}  // namespace nikolaev_r_simple_iteration_method_mpi