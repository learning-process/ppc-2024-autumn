// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalyakina_a_trapezoidal_integration_seq {

class TrapezoidalIntegrationTask : public ppc::core::Task {
  unsigned int CalculationOfCoefficient(const std::vector<double>& point);
  void Recursive(std::vector<double>& _point, unsigned int& definition, unsigned int divider, unsigned int variable);
  std::vector<double> GetPointFromNumber(unsigned int number);

 public:
  explicit TrapezoidalIntegrationTask(std::shared_ptr<ppc::core::TaskData> taskData_,
                                      double (*_function)(std::vector<double>))
      : Task(std::move(taskData_)), function(_function) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double (*function)(std::vector<double>);
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> number_of_intervals;
  double result;
};

}  // namespace kalyakina_a_trapezoidal_integration_seq