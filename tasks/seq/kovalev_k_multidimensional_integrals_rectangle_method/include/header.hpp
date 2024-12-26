#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stack>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_multidimensional_integrals_rectangle_method_seq {

class MultidimensionalIntegralsRectangleMethod : public ppc::core::Task {
 private:
  std::vector<std::pair<double, double>> limits;
  size_t n;
  std::function<double(std::vector<double>& args)> func;
  double h;
  double res;

 public:
  explicit MultidimensionalIntegralsRectangleMethod(const std::shared_ptr<ppc::core::TaskData>& taskData_,
                                                    std::function<double(std::vector<double>& args)> func_)
      : Task(taskData_), n(taskData_->inputs_count[0]), func(std::move(func_)) {}
  bool count_multidimensional_integrals_rectangle_method_seq();
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_multidimensional_integrals_rectangle_method_seq