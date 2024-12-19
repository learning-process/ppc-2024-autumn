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

namespace shvedova_v_mult_int_simpson_mpi {

using FunctionType = std::function<double(std::vector<double>&)>;
using Limits = std::deque<std::pair<double, double>>;
using Simpson1DFunction = std::function<double(double)>;

double integrateSimpsonImpl(Limits& integrationLimits, std::vector<double>& currentArguments,
                            const FunctionType& integrandFunction, double precision);

double integrateSimpson(Limits integrationLimits, double precision, const FunctionType& integrandFunction);

double integrateSimpson1D(double lowerBound, double upperBound, int segmentCount, const Simpson1DFunction& integrand);

double integrateSimpsonParallel(boost::mpi::communicator& world, Limits integrationLimits, double precision,
                                const FunctionType& integrandFunction);

class SimpsonMultIntSequential : public ppc::core::Task {
 public:
  explicit SimpsonMultIntSequential(std::shared_ptr<ppc::core::TaskData> taskData_, FunctionType& func_)
      : Task(std::move(taskData_)), integrandFunction(func_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Limits integrationLimits;
  FunctionType integrandFunction;
  double tolerance_{};
  double result_{};
};

class SimpsonMultIntParallel : public ppc::core::Task {
 public:
  explicit SimpsonMultIntParallel(std::shared_ptr<ppc::core::TaskData> taskData_, FunctionType& func_)
      : Task(std::move(taskData_)), integrandFunction(func_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Limits integrationLimits;
  FunctionType integrandFunction;
  double tolerance_{};
  double result_{};

  boost::mpi::communicator world;
};

}  // namespace shvedova_v_mult_int_simpson_mpi
