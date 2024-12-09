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

double integrateSimpsonImpl(std::deque<std::pair<double, double>>& integrationLimits,
                            std::vector<double>& currentArguments,
                            const std::function<double(std::vector<double>& args)>& integrandFunction,
                            double precision);

double integrateSimpson(std::deque<std::pair<double, double>> integrationLimits, double precision,
                        const std::function<double(std::vector<double>& args)>& integrandFunction);

double integrateSimpsonParallel(boost::mpi::communicator& world,
                                std::deque<std::pair<double, double>> integrationLimits, double precision,
                                const std::function<double(std::vector<double>& args)>& integrandFunction);

class SimpsonMultIntSequential : public ppc::core::Task {
 public:
  explicit SimpsonMultIntSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                    std::function<double(std::vector<double>& args)>& func_)
      : Task(std::move(taskData_)), integrandFunction(func_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::deque<std::pair<double, double>> integrationLimits;
  std::function<double(std::vector<double>& args)> integrandFunction;
  double tolerance_{};
  double result_{};
};

class SimpsonMultIntParallel : public ppc::core::Task {
 public:
  explicit SimpsonMultIntParallel(std::shared_ptr<ppc::core::TaskData> taskData_,
                                  std::function<double(std::vector<double>& args)>& func_)
      : Task(std::move(taskData_)), integrandFunction(func_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::deque<std::pair<double, double>> integrationLimits;
  std::function<double(std::vector<double>& args)> integrandFunction;
  double tolerance_{};
  double result_{};

  boost::mpi::communicator world;
};

}  // namespace shvedova_v_mult_int_simpson_mpi
