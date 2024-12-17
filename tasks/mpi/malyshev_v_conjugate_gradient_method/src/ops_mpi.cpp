#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

bool malyshev_conjugate_gradient_method::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  matrix_.resize(size, std::vector<double>(size));
  b_.resize(size);
  x_.resize(size, 0.0);

  double* data;
  for (uint32_t i = 0; i < size; i++) {
    data = reinterpret_cast<double*>(taskData->inputs[0]) + i * size;
    std::copy(data, data + size, matrix_[i].data());
  }

  data = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(data, data + size, b_.data());

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() != 2 || taskData->inputs_count.size() < 2) {
    return false;
  }

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_conjugate_gradient_method::TestTaskSequential::run() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];
  std::vector<double> r(size), p(size), Ap(size);
  double rsold, rsnew, alpha;

  // Initial residual
  for (uint32_t i = 0; i < size; i++) {
    r[i] = b_[i];
    for (uint32_t j = 0; j < size; j++) {
      r[i] -= matrix_[i][j] * x_[j];
    }
    p[i] = r[i];
  }

  rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

  for (uint32_t k = 0; k < size; k++) {
    // Compute Ap = A * p
    for (uint32_t i = 0; i < size; i++) {
      Ap[i] = 0.0;
      for (uint32_t j = 0; j < size; j++) {
        Ap[i] += matrix_[i][j] * p[j];
      }
    }

    // Compute alpha = rsold / (p' * Ap)
    alpha = rsold / std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

    // Update x and r
    for (uint32_t i = 0; i < size; i++) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    rsnew = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    if (std::sqrt(rsnew) < 1e-10) break;

    // Update p
    for (uint32_t i = 0; i < size; i++) {
      p[i] = r[i] + (rsnew / rsold) * p[i];
    }

    rsold = rsnew;
  }

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(x_.begin(), x_.end(), reinterpret_cast<double*>(taskData->outputs[0]));

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t size = taskData->inputs_count[0];

    delta_ = size / world.size();
    ext_ = size % world.size();

    matrix_.resize(size, std::vector<double>(size));
    b_.resize(size);
    x_.resize(size, 0.0);

    double* data;
    for (uint32_t i = 0; i < size; i++) {
      data = reinterpret_cast<double*>(taskData->inputs[0]) + i * size;
      std::copy(data, data + size, matrix_[i].data());
    }

    data = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(data, data + size, b_.data());
  }

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.size() != 2 || taskData->inputs_count.size() < 2) {
      return false;
    }

    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);
  broadcast(world, b_, 0);

  std::vector<int> sizes(world.size(), delta_);
  for (uint32_t i = 0; i < ext_; i++) {
    sizes[world.size() - i - 1]++;
  }

  local_matrix_.resize(sizes[world.rank()]);
  local_x_.resize(sizes[world.rank()], 0.0);

  scatterv(world, matrix_, sizes, local_matrix_.data(), 0);

  std::vector<double> r(sizes[world.rank()]), p(sizes[world.rank()]), Ap(sizes[world.rank()]);
  double rsold, rsnew, alpha;

  // Initial residual
  for (uint32_t i = 0; i < static_cast<uint32_t>(sizes[world.rank()]); i++) {
    r[i] = b_[i];
    for (uint32_t j = 0; j < static_cast<uint32_t>(sizes[world.rank()]); j++) {
      r[i] -= local_matrix_[i][j] * local_x_[j];
    }
    p[i] = r[i];
  }

  rsold = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

  for (uint32_t k = 0; k < static_cast<uint32_t>(sizes[world.rank()]); k++) {
    // Compute Ap = A * p
    for (uint32_t i = 0; i < static_cast<uint32_t>(sizes[world.rank()]); i++) {
      Ap[i] = 0.0;
      for (uint32_t j = 0; j < static_cast<uint32_t>(sizes[world.rank()]); j++) {
        Ap[i] += local_matrix_[i][j] * p[j];
      }
    }

    // Compute alpha = rsold / (p' * Ap)
    alpha = rsold / std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

    // Update x and r
    for (uint32_t i = 0; i < static_cast<uint32_t>(sizes[world.rank()]); i++) {
      local_x_[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    rsnew = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    if (std::sqrt(rsnew) < 1e-10) break;

    // Update p
    for (uint32_t i = 0; i < static_cast<uint32_t>(sizes[world.rank()]); i++) {
      p[i] = r[i] + (rsnew / rsold) * p[i];
    }

    rsold = rsnew;
  }

  gatherv(world, local_x_, x_.data(), sizes, 0);

  return true;
}

bool malyshev_conjugate_gradient_method::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(x_.begin(), x_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}