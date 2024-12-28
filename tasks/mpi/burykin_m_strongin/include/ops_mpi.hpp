#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_strongin {

class StronginSequential : public ppc::core::Task {
 public:
  explicit StronginSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double)> f_)
      : Task(std::move(taskData_)), f(std::move(f_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double res{0};
  double x0{0}, x1{0}, eps{0};
  std::function<double(double)> f;
};

class StronginParallel : public ppc::core::Task {
 public:
  explicit StronginParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double)> f_)
      : Task(std::move(taskData_)), f(std::move(f_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double res{0};
  double x0{0}, x1{0}, eps{0};
  std::function<double(double)> f;
  boost::mpi::communicator world;
};

}  // namespace burykin_m_strongin