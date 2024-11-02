#pragma once
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_monte_carlo_integration {

class MonteCarloIntegration : public ppc::core::Task {
 public:
  explicit MonteCarloIntegration(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double function_to_integrate(double x);  

 private:
  double a, b;
  int num_points; 
  double result{};
  std::mt19937 rng;
  std::uniform_real_distribution<double> dist;

  double monte_carlo_integration();
};

}  // namespace malyshev_v_monte_carlo_integration
