#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_monte_carlo {

class MonteCarloIntegrationTaskSequential : public ppc::core::Task {
 public:
  explicit MonteCarloIntegrationTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_interval(double a, double b) {
    a_ = a;
    b_ = b;
  }

  void set_function(std::function<double(double)> func) { f_ = std::move(func); }

  void set_num_points(int num_points) { num_points_ = num_points; }

 private:
  double integral_value_{};
  double a_ = 0.0;
  double b_ = 1.0;
  int num_points_ = 1000000;
  std::function<double(double)> f_ = [](double x) { return x * x; };
};

}  // namespace shuravina_o_monte_carlo