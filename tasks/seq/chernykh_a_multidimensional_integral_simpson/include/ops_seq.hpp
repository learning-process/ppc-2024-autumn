#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_simpson_seq {

using func_nd_t = std::function<double(const std::vector<double>&)>;
using bounds_t = std::vector<std::pair<double, double>>;
using steps_t = std::vector<int>;

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> task_data, func_nd_t& func)
      : Task(std::move(task_data)), func(func) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  func_nd_t func;
  bounds_t bounds;
  steps_t steps;
  double result{};

  std::vector<double> get_step_sizes() const;
  int get_total_points() const;
  std::vector<int> get_indices(int p) const;
  std::vector<double> get_point(std::vector<int>& indices, std::vector<double>& step_sizes) const;
  double get_weight(std::vector<int>& indices) const;
  double get_scaling_factor(std::vector<double>& step_sizes) const;
};

}  // namespace chernykh_a_multidimensional_integral_simpson_seq
