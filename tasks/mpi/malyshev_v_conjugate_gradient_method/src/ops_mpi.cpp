#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

bool malyshev_v_conjugate_gradient_method::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  matrix_.resize(size, std::vector<double>(size));
  vector_.resize(size);
  result_.resize(size);

  double* data;
  for (uint32_t i = 0; i < matrix_.size(); i++) {
    data = reinterpret_cast<double*>(taskData->inputs[i]);
    std::copy(data, data + size, matrix_[i].data());
  }

  data = reinterpret_cast<double*>(taskData->inputs[size]);
  std::copy(data, data + size, vector_.data());

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskSequential::validation() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  if (taskData->inputs.size() != size + 1 || taskData->inputs_count.size() < 2) {
    return false;
  }

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_v_conjugate_gradient_method::TestTaskSequential::run() {
  internal_order_test();

  uint32_t size = matrix_.size();
  std::vector<double> residual(size);
  std::vector<double> direction(size);
  std::vector<double> temp(size);

  std::fill(result_.begin(), result_.end(), 0.0);

  for (uint32_t i = 0; i < size; ++i) {
    residual[i] = vector_[i];
    for (uint32_t j = 0; j < size; ++j) {
      residual[i] -= matrix_[i][j] * result_[j];
    }
    direction[i] = residual[i];
  }

  double residual_norm_sq = 0.0;
  for (uint32_t i = 0; i < size; ++i) {
    residual_norm_sq += residual[i] * residual[i];
  }

  const double tolerance = 1e-6;
  const uint32_t max_iterations = size;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    std::fill(temp.begin(), temp.end(), 0.0);
    for (uint32_t i = 0; i < size; ++i) {
      for (uint32_t j = 0; j < size; ++j) {
        temp[i] += matrix_[i][j] * direction[j];
      }
    }

    double alpha_numerator = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
      alpha_numerator += residual[i] * residual[i];
    }

    double alpha_denominator = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
      alpha_denominator += direction[i] * temp[i];
    }

    if (alpha_denominator == 0.0) {
      break;
    }

    double alpha = alpha_numerator / alpha_denominator;

    for (uint32_t i = 0; i < size; ++i) {
      result_[i] += alpha * direction[i];
    }

    for (uint32_t i = 0; i < size; ++i) {
      residual[i] -= alpha * temp[i];
    }

    double new_residual_norm_sq = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
      new_residual_norm_sq += residual[i] * residual[i];
    }

    if (std::sqrt(new_residual_norm_sq) < tolerance) {
      break;
    }

    double beta = new_residual_norm_sq / residual_norm_sq;
    residual_norm_sq = new_residual_norm_sq;

    for (uint32_t i = 0; i < size; ++i) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t size = taskData->inputs_count[0];

    delta_ = size / world.size();
    ext_ = size % world.size();

    matrix_.resize(size, std::vector<double>(size));
    vector_.resize(size);
    result_.resize(size);

    double* data;
    for (uint32_t i = 0; i < matrix_.size(); i++) {
      data = reinterpret_cast<double*>(taskData->inputs[i]);
      std::copy(data, data + size, matrix_[i].data());
    }

    data = reinterpret_cast<double*>(taskData->inputs[size]);
    std::copy(data, data + size, vector_.data());
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t size = taskData->inputs_count[0];

    if (taskData->inputs.size() != size + 1 || taskData->inputs_count.size() < 2) {
      return false;
    }

    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);
  broadcast(world, vector_, 0);

  std::vector<int> sizes(world.size(), delta_);
  for (uint32_t i = 0; i < ext_; i++) {
    sizes[world.size() - i - 1]++;
  }

  local_matrix_.resize(sizes[world.rank()]);
  local_result_.resize(sizes[world.rank()]);

  scatterv(world, matrix_, sizes, local_matrix_.data(), 0);

  uint32_t local_size = local_matrix_.size();
  std::vector<double> local_residual(local_size);
  std::vector<double> local_direction(local_size);
  std::vector<double> local_temp(local_size);

  std::fill(local_result_.begin(), local_result_.end(), 0.0);

  for (uint32_t i = 0; i < local_size; ++i) {
    local_residual[i] = vector_[i];
    for (uint32_t j = 0; j < local_size; ++j) {
      local_residual[i] -= local_matrix_[i][j] * local_result_[j];
    }
    local_direction[i] = local_residual[i];
  }

  double local_residual_norm_sq = 0.0;
  for (uint32_t i = 0; i < local_size; ++i) {
    local_residual_norm_sq += local_residual[i] * local_residual[i];
  }

  double global_residual_norm_sq;
  reduce(world, local_residual_norm_sq, global_residual_norm_sq, std::plus<>(), 0);
  broadcast(world, global_residual_norm_sq, 0);

  const double tolerance = 1e-6;
  const uint32_t max_iterations = local_size;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    std::fill(local_temp.begin(), local_temp.end(), 0.0);
    for (uint32_t i = 0; i < local_size; ++i) {
      for (uint32_t j = 0; j < local_size; ++j) {
        local_temp[i] += local_matrix_[i][j] * local_direction[j];
      }
    }

    double local_alpha_numerator = 0.0;
    for (uint32_t i = 0; i < local_size; ++i) {
      local_alpha_numerator += local_residual[i] * local_residual[i];
    }

    double global_alpha_numerator;
    reduce(world, local_alpha_numerator, global_alpha_numerator, std::plus<>(), 0);
    broadcast(world, global_alpha_numerator, 0);

    double local_alpha_denominator = 0.0;
    for (uint32_t i = 0; i < local_size; ++i) {
      local_alpha_denominator += local_direction[i] * local_temp[i];
    }

    double global_alpha_denominator;
    reduce(world, local_alpha_denominator, global_alpha_denominator, std::plus<>(), 0);
    broadcast(world, global_alpha_denominator, 0);

    if (global_alpha_denominator == 0.0) {
      break;
    }

    double global_alpha = global_alpha_numerator / global_alpha_denominator;

    for (uint32_t i = 0; i < local_size; ++i) {
      local_result_[i] += global_alpha * local_direction[i];
    }

    for (uint32_t i = 0; i < local_size; ++i) {
      local_residual[i] -= global_alpha * local_temp[i];
    }

    double new_local_residual_norm_sq = 0.0;
    for (uint32_t i = 0; i < local_size; ++i) {
      new_local_residual_norm_sq += local_residual[i] * local_residual[i];
    }

    double new_global_residual_norm_sq;
    reduce(world, new_local_residual_norm_sq, new_global_residual_norm_sq, std::plus<>(), 0);
    broadcast(world, new_global_residual_norm_sq, 0);

    if (std::sqrt(new_global_residual_norm_sq) < tolerance) {
      break;
    }

    double beta = new_global_residual_norm_sq / global_residual_norm_sq;
    global_residual_norm_sq = new_global_residual_norm_sq;

    for (uint32_t i = 0; i < local_size; ++i) {
      local_direction[i] = local_residual[i] + beta * local_direction[i];
    }
  }

  gatherv(world, local_result_, result_.data(), sizes, 0);

  return true;
}

bool malyshev_v_conjugate_gradient_method::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}