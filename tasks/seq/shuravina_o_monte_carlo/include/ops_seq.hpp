#pragma once

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

 private:
  double integral_value_{};
};

}  // namespace shuravina_o_monte_carlo