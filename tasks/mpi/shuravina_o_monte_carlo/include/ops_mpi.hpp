#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_monte_carlo {

class MonteCarloIntegrationTaskParallel : public ppc::core::Task {
 public:
  explicit MonteCarloIntegrationTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double get_integral_value() const { return integral_value_; }

 private:
  double integral_value_{};
  boost::mpi::communicator world;
};
}  // namespace shuravina_o_monte_carlo