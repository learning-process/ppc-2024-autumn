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

namespace lysov_i_integration_the_trapezoid_method_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a = 0.0;
  double b = 0.0;
  double h = 0.0;
  int cnt_of_splits = 0;
  double local_a;
  int local_cnt_of_splits;
  double static function_square(double x) { return x * x; }

 private:
  std::vector<int> input_;
  int res{};
  std::string ops;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a = 0.0;
  double b = 0.0;
  double h = 0.0;
  int cnt_of_splits = 0;
  double local_a;
  int local_cnt_of_splits;
  static double function_square(double x) { return x * x; }

 private:
  std::vector<int> input_, local_input_;
  double res;
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace lysov_i_integration_the_trapezoid_method_mpi