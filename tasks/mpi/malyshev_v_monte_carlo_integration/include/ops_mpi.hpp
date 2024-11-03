#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_monte_carlo_integration {

class MonteCarloIntegrationSequential : public ppc::core::Task {
 public:
  explicit MonteCarloIntegrationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double a, double b, int n);
  double a_{};
  double b_{};
  int n_{};
  double result_{};
  std::function<double(double)> func_;
};

class MonteCarloIntegrationParallel : public ppc::core::Task {
 public:
  explicit MonteCarloIntegrationParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  double parallel_integrate(const std::function<double(double)>& f, double a, double b, int n);

  double a_{};
  double b_{};
  int n_{};
  double global_result_{};
  std::function<double(double)> func_;

  boost::mpi::communicator world;
};

}  // namespace malyshev_v_monte_carlo_integration