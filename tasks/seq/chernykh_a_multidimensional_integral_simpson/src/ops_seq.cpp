#include "seq/chernykh_a_multidimensional_integral_simpson/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace chernykh_a_multidimensional_integral_simpson_seq {

bool SequentialTask::validation() {
  internal_order_test();
  auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[0]);
  auto bounds_size = taskData->inputs_count[0];
  auto *steps_ptr = reinterpret_cast<int *>(taskData->inputs[1]);
  auto steps_size = taskData->inputs_count[1];

  auto correct_bounds = std::all_of(bounds_ptr, bounds_ptr + bounds_size, [](auto &b) { return b.first < b.second; });
  auto correct_steps = std::all_of(steps_ptr, steps_ptr + steps_size, [](auto &s) { return s > 0 && s % 2 == 0; });

  return bounds_size > 0 && correct_bounds && steps_size > 0 && correct_steps && bounds_size == steps_size;
}

bool SequentialTask::pre_processing() {
  internal_order_test();
  auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[0]);
  auto bounds_size = taskData->inputs_count[0];
  auto *steps_ptr = reinterpret_cast<int *>(taskData->inputs[1]);
  auto steps_size = taskData->inputs_count[1];

  bounds.assign(bounds_ptr, bounds_ptr + bounds_size);
  steps.assign(steps_ptr, steps_ptr + steps_size);
  return true;
}

bool SequentialTask::run() {
  internal_order_test();
  auto step_sizes = get_step_sizes();
  auto total_points = get_total_points();

  auto sum = 0.0;
  for (int p = 0; p < total_points; p++) {
    auto indices = get_indices(p);
    auto point = get_point(indices, step_sizes);
    auto weight = get_weight(indices);

    sum += weight * func(point);
  }

  result = sum * get_scaling_factor(step_sizes);

  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<double *>(taskData->outputs[0]) = result;
  return true;
}

std::vector<double> SequentialTask::get_step_sizes() const {
  std::vector<double> step_sizes(bounds.size());
  for (size_t i = 0; i < bounds.size(); i++) {
    step_sizes[i] = (bounds[i].second - bounds[i].first) / steps[i];
  }
  return step_sizes;
}

int SequentialTask::get_total_points() const {
  int total_points = 1;
  for (size_t i = 0; i < bounds.size(); i++) {
    total_points *= steps[i] + 1;
  }
  return total_points;
}

std::vector<int> SequentialTask::get_indices(int p) const {
  std::vector<int> indices(bounds.size());
  for (size_t i = 0; i < bounds.size(); i++) {
    indices[i] = p % (steps[i] + 1);
    p /= (steps[i] + 1);
  }
  return indices;
}

std::vector<double> SequentialTask::get_point(std::vector<int> &indices, std::vector<double> &step_sizes) const {
  std::vector<double> point(bounds.size());
  for (size_t i = 0; i < bounds.size(); i++) {
    point[i] = bounds[i].first + indices[i] * step_sizes[i];
  }
  return point;
}

double SequentialTask::get_weight(std::vector<int> &indices) const {
  double weight = 1.0;
  for (size_t i = 0; i < bounds.size(); i++) {
    if (indices[i] == 0 || indices[i] == steps[i]) {
      weight *= 1.0;
    } else if (indices[i] % 2 != 0) {
      weight *= 4.0;
    } else {
      weight *= 2.0;
    }
  }
  return weight;
}

double SequentialTask::get_scaling_factor(std::vector<double> &step_sizes) const {
  double scaling_factor = 1.0;
  for (size_t i = 0; i < bounds.size(); i++) {
    scaling_factor *= step_sizes[i] / 3.0;
  }
  return scaling_factor;
}

}  // namespace chernykh_a_multidimensional_integral_simpson_seq
