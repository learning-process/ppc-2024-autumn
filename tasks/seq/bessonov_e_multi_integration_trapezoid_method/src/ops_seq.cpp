#include "seq/bessonov_e_multi_integration_trapezoid_method/include/ops_seq.hpp"

std::vector<double> bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::precompute_weights(
    size_t dimensions) {
  size_t combinations = static_cast<size_t>(1) << dimensions;  // 2^dimensions
  std::vector<double> weights(combinations);

  for (size_t mask = 0; mask < combinations; ++mask) {
    double weight = 1.0;
    for (size_t i = 0; i < dimensions; ++i) {
      if ((mask & (static_cast<size_t>(1) << i)) != 0) {
        weight *= 0.5;
      }
    }
    weights[mask] = weight;
  }

  return weights;
}

double bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::compute_weight_for_point(
    const std::vector<double>& point) {
  size_t mask = 0;
  for (size_t i = 0; i < dim; ++i) {
    if (point[i] == lower_bounds[i] || point[i] == upper_bounds[i]) {
      mask |= (static_cast<size_t>(1) << i);
    }
  }
  return cached_weights[mask];
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() < 3) {
    return false;
  }

  if (taskData->outputs.size() != 1) {
    return false;
  }

  auto* dim_ptr = reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (dim_ptr == nullptr || *dim_ptr == 0) {
    return false;
  }

  auto* lower_data = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* upper_data = reinterpret_cast<double*>(taskData->inputs[2]);
  auto* steps_data = reinterpret_cast<int*>(taskData->inputs[3]);

  if (lower_data == nullptr || upper_data == nullptr || steps_data == nullptr) {
    return false;
  }

  if (taskData->inputs_count.size() < 3 || taskData->inputs_count[0] != *dim_ptr ||
      taskData->inputs_count[1] != *dim_ptr || taskData->inputs_count[2] != *dim_ptr) {
    return false;
  }

  for (size_t i = 0; i < *dim_ptr; ++i) {
    if (lower_data[i] >= upper_data[i]) {
      return false;
    }
    if (steps_data[i] <= 0) {
      return false;
    }
  }

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  dim = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  lower_bounds.assign(reinterpret_cast<double*>(taskData->inputs[1]),
                      reinterpret_cast<double*>(taskData->inputs[1]) + dim);
  upper_bounds.assign(reinterpret_cast<double*>(taskData->inputs[2]),
                      reinterpret_cast<double*>(taskData->inputs[2]) + dim);
  num_steps.assign(reinterpret_cast<int*>(taskData->inputs[3]), reinterpret_cast<int*>(taskData->inputs[3]) + dim);

  cached_weights = precompute_weights(dim);

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::run() {
  internal_order_test();

  std::vector<double> step_sizes(dim);
  for (size_t i = 0; i < dim; ++i) {
    step_sizes[i] = (upper_bounds[i] - lower_bounds[i]) / num_steps[i];
  }

  size_t total_points = 1;
  for (int steps : num_steps) {
    total_points *= (steps + 1);
  }

  result = 0.0;
  std::vector<double> point(dim);

  for (size_t i = 0; i < total_points; ++i) {
    size_t temp = i;

    for (size_t j = 0; j < dim; ++j) {
      point[j] = lower_bounds[j] + (temp % (num_steps[j] + 1)) * step_sizes[j];
      temp /= (num_steps[j] + 1);
    }

    double weight = compute_weight_for_point(point);
    result += integrand(point) * weight;
  }

  result *= std::accumulate(step_sizes.begin(), step_sizes.end(), 1.0, std::multiplies<>());

  return true;
}

bool bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}