#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_strongin {

class StronginOptimization : public ppc::core::Task {
 public:
  explicit StronginOptimization(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  static double f(double x);

  static double strongin_method(double x0, double x1, double eps);

  double result = 0.0;
  double x0 = 0.0, x1 = 0.0, epsilon = 0.0001;  // Параметры
};

}  // namespace burykin_m_strongin
