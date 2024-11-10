#pragma once

#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_monte_carlo {

class MonteCarloIntegrationTaskParallel : public ppc::core::Task {
 public:
  explicit MonteCarloIntegrationTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), gen(rd()), dis_(a_, b_) {}

  void set_interval(double a, double b) {
    a_ = a;
    b_ = b;
    dis_ = std::uniform_real_distribution<>(a_, b_);
  }

  void set_num_points(int num_points) { num_points_ = num_points; }

  void set_function(std::function<double(double)> func) { f_ = std::move(func); }

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double get_integral_value() const { return integral_value_; }

 private:
  double integral_value_{};
  double a_ = 0.0;
  double b_ = 1.0;
  int num_points_ = 1000000;
  std::function<double(double)> f_ = [](double x) { return x * x; };
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> dis_;
};

}  // namespace shuravina_o_monte_carlo