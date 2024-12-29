#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

bool malyshev_conjugate_gradient::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  matrix_.resize(size, std::vector<double>(size));
  vector_.resize(size);
  result_.resize(size);

  auto* data = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(data, data + size * size, matrix_[0].data());

  data = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(data, data + size, vector_.data());

  return true;
}

bool malyshev_conjugate_gradient::TestTaskSequential::validation() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  if (taskData->inputs.size() != 2 || taskData->inputs_count.empty()) {
    return false;
  }

  return taskData->outputs_count[0] == size;
}

bool malyshev_conjugate_gradient::TestTaskSequential::run() {
  internal_order_test();

  uint32_t size = vector_.size();
  std::vector<double> x(size, 0.0);
  std::vector<double> r = vector_;
  std::vector<double> p = r;
  double rsold = 0.0;

  for (uint32_t i = 0; i < size; ++i) {
    rsold += r[i] * r[i];
  }

  const uint32_t maxIterations = 1000;
  uint32_t iteration = 0;

  for (iteration = 0; iteration < maxIterations; ++iteration) {
    std::vector<double> Ap(size, 0.0);
    for (uint32_t j = 0; j < size; ++j) {
      for (uint32_t k = 0; k < size; ++k) {
        Ap[j] += matrix_[j * size + k] * p[k];
      }
    }

    double pAp = 0.0;
    for (uint32_t j = 0; j < size; ++j) {
      pAp += p[j] * Ap[j];
    }

    if (std::abs(pAp) < 1e-12) {
      std::cerr << "Error: Division by near-zero in conjugate gradient. pAp = " << pAp << std::endl;
      return false;
    }

    double alpha = rsold / pAp;
    for (uint32_t j = 0; j < size; ++j) {
      x[j] += alpha * p[j];
      r[j] -= alpha * Ap[j];
    }

    double rsnew = 0.0;
    for (uint32_t j = 0; j < size; ++j) {
      rsnew += r[j] * r[j];
    }

    if (sqrt(rsnew) < 1e-6) {
      break;
    }

    for (uint32_t j = 0; j < size; ++j) {
      p[j] = r[j] + (rsnew / rsold) * p[j];
    }

    rsold = rsnew;
  }

  if (iteration == maxIterations) {
    std::cerr << "Conjugate gradient method did not converge within the maximum number of iterations." << std::endl;
    return false;
  }

  result_ = x;

  return true;
}

bool malyshev_conjugate_gradient::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));

  return true;
}

bool malyshev_conjugate_gradient::TestTaskParallel::run() {
  internal_order_test();

  uint32_t size = vector_.size();
  std::vector<double> x(size, 0.0);
  std::vector<double> r(size, 0.0);
  std::vector<double> p(size, 0.0);
  std::vector<double> Ap(size, 0.0);
  double rsold = 0.0;

  if (world.rank() == 0) {
    r = vector_;
    p = r;
    for (uint32_t i = 0; i < size; ++i) {
      rsold += r[i] * r[i];
    }
  }

  broadcast(world, rsold, 0);

  const uint32_t maxIterations = 1000;
  uint32_t iteration = 0;

  for (iteration = 0; iteration < maxIterations; ++iteration) {
    if (world.rank() == 0) {
      for (uint32_t j = 0; j < size; ++j) {
        Ap[j] = 0.0;
        for (uint32_t k = 0; k < size; ++k) {
          Ap[j] += matrix_[j * size + k] * p[k];
        }
      }
    }

    broadcast(world, Ap, 0);

    double local_pAp = 0.0;
    for (uint32_t j = 0; j < size; ++j) {
      local_pAp += p[j] * Ap[j];
    }

    double global_pAp = 0.0;
    reduce(world, local_pAp, global_pAp, std::plus<>(), 0);

    if (world.rank() == 0 && std::abs(global_pAp) < 1e-12) {
      std::cerr << "Error: Division by near-zero in conjugate gradient. global_pAp = " << global_pAp << std::endl;
      return false;
    }

    double alpha = 0.0;
    if (world.rank() == 0) {
      alpha = rsold / global_pAp;
      for (uint32_t j = 0; j < size; ++j) {
        x[j] += alpha * p[j];
        r[j] -= alpha * Ap[j];
      }
    }

    broadcast(world, r, 0);

    double rsnew = 0.0;
    if (world.rank() == 0) {
      for (uint32_t j = 0; j < size; ++j) {
        rsnew += r[j] * r[j];
      }
    }

    broadcast(world, rsnew, 0);

    if (sqrt(rsnew) < 1e-6) {
      if (world.rank() == 0) {
        result_ = x;
      }
      break;
    }

    if (world.rank() == 0) {
      for (uint32_t j = 0; j < size; ++j) {
        p[j] = r[j] + (rsnew / rsold) * p[j];
      }
    }

    broadcast(world, p, 0);
    rsold = rsnew;

    if (world.rank() == 0) {
      std::cerr << "Iteration " << iteration << ": Residual norm = " << sqrt(rsnew) << std::endl;
    }
  }

  if (iteration == maxIterations) {
    if (world.rank() == 0) {
      std::cerr << "Max iterations reached. Exiting." << std::endl;
    }
    return false;
  }

  return iteration < maxIterations;
}

bool malyshev_conjugate_gradient::TestTaskParallel::pre_processing() {
  double det = 1.0;
  uint32_t size = vector_.size();

  for (uint32_t i = 0; i < size; ++i) {
    det *= matrix_[i * size + i];
  }

  if (std::abs(det) < 1e-12) {
    std::cerr << "Matrix is singular or near-singular. No unique solution." << std::endl;
    return false;
  }

  return true;
}

bool malyshev_conjugate_gradient::TestTaskParallel::validation() {
  if (matrix_.empty() || vector_.empty()) {
    std::cerr << "Error: Matrix or vector is empty." << std::endl;
    return false;
  }

  uint32_t size = vector_.size();
  if (matrix_.size() != size * size) {
    std::cerr << "Error: Matrix size does not match vector size." << std::endl;
    return false;
  }

  return true;
}

bool malyshev_conjugate_gradient::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}