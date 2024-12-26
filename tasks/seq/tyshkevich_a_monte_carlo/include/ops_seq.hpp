#pragma once

#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tyshkevich_a_monte_carlo_seq {

class MonteCarloSequential : public ppc::core::Task {
 public:
  explicit MonteCarloSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                std::function<double(const std::vector<double>&)> func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double>&)> func;
  int dimensions;
  double precision, globalSum, result = 0.0;

  std::vector<std::pair<double, double>> bounds;
  std::mt19937 gen;
  std::vector<std::uniform_real_distribution<double>> distributions;
};

}  // namespace tyshkevich_a_monte_carlo_seq