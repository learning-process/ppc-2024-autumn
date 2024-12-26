// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/utility.hpp>
#include <cmath>
#include <deque>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_mpi {
using function = std::function<double(std::vector<double>& args)>;

double integrateImpl(std::deque<std::pair<double, double>>& limits, std::vector<double>& args, const function& func,
                     double eps);
double integrateSeq(std::deque<std::pair<double, double>> limits, double eps, const function& func);
double integrateMPI(boost::mpi::communicator& world, std::deque<std::pair<double, double>> limits, double eps,
                    const function& func);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, function& func_)
      : Task(std::move(taskData_)), function_(func_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::deque<std::pair<double, double>> limits_;
  function function_;
  double eps_{};
  double res_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, function& func_)
      : Task(std::move(taskData_)), function_(func_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::deque<std::pair<double, double>> limits_;
  function function_;
  double eps_{};
  double res_{};

  boost::mpi::communicator world;
};

}  // namespace ermolaev_v_multidimensional_integral_rectangle_mpi