#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

#include "core/task/include/task.hpp"

namespace shulpin_monte_carlo_integration {
using func = std::function<double(double)>;

double fsin(double x);
double fcos(double x);
double f_two_sin_cos(double x);

double integral(double a, double b, int N, const func &f);

double parallel_integral(double a, double b, int N, const func &f);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(const std::shared_ptr<ppc::core::TaskData> &taskData_)
      : Task(taskData_ ? taskData_ : std::make_shared<ppc::core::TaskData>()) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_seq(const func &f);

 private:
  double a_seq{};
  double b_seq{};
  double N_seq{};
  func func_seq;
  double res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(const std::shared_ptr<ppc::core::TaskData> &taskData_)
      : Task(taskData_ ? taskData_ : std::make_shared<ppc::core::TaskData>()) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_MPI(const func &f);

 private:
  double a_MPI{};
  double b_MPI{};
  int N_MPI{};
  func func_MPI;
  double res{};
  boost::mpi::communicator world;
};
}  // namespace shulpin_monte_carlo_integration