#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <vector>
#include <cmath>

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  matrix_.resize(size, std::vector<double>(size));
  vector_.resize(size);
  solution_.resize(size);

  double* data;
  for (uint32_t i = 0; i < matrix_.size(); i++) {
    data = reinterpret_cast<double*>(taskData->inputs[0]) + i * size;
    std::copy(data, data + size, matrix_[i].data());
  }

  data = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(data, data + size, vector_.data());

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::run() {
  internal_order_test();

  uint32_t size = matrix_.size();
  std::vector<double> x(size, 0.0);
  std::vector<double> r = vector_;
  std::vector<double> p = r;
  double rsold = 0.0;
  for (uint32_t i = 0; i < size; i++) {
    rsold += r[i] * r[i];
  }

  for (uint32_t iter = 0; iter < size; iter++) {
    std::vector<double> Ap(size, 0.0);
    for (uint32_t i = 0; i < size; i++) {
      for (uint32_t j = 0; j < size; j++) {
        Ap[i] += matrix_[i][j] * p[j];
      }
    }

    double alpha = rsold / std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

    for (uint32_t i = 0; i < size; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rsnew = 0.0;
    for (uint32_t i = 0; i < size; i++) {
      rsnew += r[i] * r[i];
    }

    if (std::sqrt(rsnew) < 1e-10) {
      break;
    }

    for (uint32_t i = 0; i < size; i++) {
      p[i] = r[i] + (rsnew / rsold) * p[i];
    }

    rsold = rsnew;
  }

  solution_ = x;

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(solution_.begin(), solution_.end(), reinterpret_cast<double*>(taskData->outputs[0]));

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    uint32_t size = taskData->inputs_count[0];

    delta_ = size / world.size();
    ext_ = size % world.size();

    matrix_.resize(size, std::vector<double>(size));
    vector_.resize(size);
    solution_.resize(size);

    double* data;
    for (uint32_t i = 0; i < matrix_.size(); i++) {
      data = reinterpret_cast<double*>(taskData->inputs[0]) + i * size;
      std::copy(data, data + size, matrix_[i].data());
    }

    data = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(data, data + size, vector_.data());
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, delta_, 0);
  broadcast(world, ext_, 0);

  std::vector<int32_t> sizes(world.size(), delta_);
  for (uint32_t i = 0; i < ext_; i++) {
    sizes[world.size() - i - 1]++;
  }

  local_matrix_.resize(sizes[world.rank()], std::vector<double>(matrix_.size()));
  local_vector_.resize(sizes[world.rank()]);
  local_solution_.resize(sizes[world.rank()]);

  scatterv(world, matrix_, sizes, local_matrix_.data(), 0);
  scatterv(world, vector_, sizes, local_vector_.data(), 0);

  uint32_t local_size = local_matrix_.size();
  uint32_t global_size = matrix_.size();
  std::vector<double> local_x(local_size, 0.0);
  std::vector<double> local_r = local_vector_;
  std::vector<double> local_p = local_r;
  double local_rsold = 0.0;
  for (uint32_t i = 0; i < local_size; i++) {
    local_rsold += local_r[i] * local_r[i];
  }

  double global_rsold;
  reduce(world, local_rsold, global_rsold, std::plus<double>(), 0);

  for (uint32_t iter = 0; iter < global_size; iter++) {
    std::vector<double> local_Ap(local_size, 0.0);
    for (uint32_t i = 0; i < local_size; i++) {
      for (uint32_t j = 0; j < global_size; j++) {
        local_Ap[i] += local_matrix_[i][j] * local_p[j];
      }
    }

    double local_alpha = local_rsold / std::inner_product(local_p.begin(), local_p.end(), local_Ap.begin(), 0.0);

    for (uint32_t i = 0; i < local_size; i++) {
      local_x[i] += local_alpha * local_p[i];
      local_r[i] -= local_alpha * local_Ap[i];
    }

    double local_rsnew = 0.0;
    for (uint32_t i = 0; i < local_size; i++) {
      local_rsnew += local_r[i] * local_r[i];
    }

    double global_rsnew;
    reduce(world, local_rsnew, global_rsnew, std::plus<double>(), 0);

    if (std::sqrt(global_rsnew) < 1e-10) {
      break;
    }

    for (uint32_t i = 0; i < local_size; i++) {
      local_p[i] = local_r[i] + (local_rsnew / local_rsold) * local_p[i];
    }

    local_rsold = local_rsnew;
  }

  gatherv(world, local_x, solution_.data(), sizes, 0);

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(solution_.begin(), solution_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}