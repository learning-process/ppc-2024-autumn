#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

namespace malyshev_v_conjugate_gradient_method {

bool TestTaskSequential::pre_processing() {
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

bool TestTaskSequential::validation() {
  internal_order_test();

  uint32_t size = taskData->inputs_count[0];

  if (taskData->inputs.size() != size + 1 || taskData->inputs_count.size() < 2) {
    return false;
  }

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool TestTaskSequential::run() {
  internal_order_test();

  uint32_t size = matrix_.size();
  std::vector<double> r(size), p(size), Ap(size);
  result_.assign(size, 0.0);

  // Initial residual
  for (uint32_t i = 0; i < size; ++i) {
    r[i] = vector_[i];
    for (uint32_t j = 0; j < size; ++j) {
      r[i] -= matrix_[i][j] * result_[j];
    }
  }

  p.assign(r.begin(), r.end());

  double rTr = dot(r, r);
  double tolerance = 1e-10;
  uint32_t max_iterations = size;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    // Compute Ap
    for (uint32_t i = 0; i < size; ++i) {
      Ap[i] = 0.0;
      for (uint32_t j = 0; j < size; ++j) {
        Ap[i] += matrix_[i][j] * p[j];
      }
    }

    // Compute alpha
    double pAp = dot(p, Ap);
    double alpha = rTr / pAp;

    // Update solution
    for (uint32_t i = 0; i < size; ++i) {
      result_[i] += alpha * p[i];
    }

    // Update residual
    for (uint32_t i = 0; i < size; ++i) {
      r[i] -= alpha * Ap[i];
    }

    // Check convergence
    double new_rTr = dot(r, r);
    if (std::sqrt(new_rTr) < tolerance) {
      break;
    }

    // Compute beta
    double beta = new_rTr / rTr;
    rTr = new_rTr;

    // Update search direction
    for (uint32_t i = 0; i < size; ++i) {
      p[i] = r[i] + beta * p[i];
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));

  return true;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

bool TestTaskParallel::pre_processing() {
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

bool TestTaskParallel::validation() {
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

bool TestTaskParallel::run() {
  internal_order_test();

  uint32_t size = matrix_.size();
  std::vector<double> r(size), p(size), Ap(size);
  result_.assign(size, 0.0);

  // Initial residual
  for (uint32_t i = 0; i < size; ++i) {
    r[i] = vector_[i];
    for (uint32_t j = 0; j < size; ++j) {
      r[i] -= matrix_[i][j] * result_[j];
    }
  }

  p.assign(r.begin(), r.end());

  double rTr = dot(r, r);
  double tolerance = 1e-10;
  uint32_t max_iterations = size;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    // Compute Ap
    for (uint32_t i = 0; i < size; ++i) {
      Ap[i] = 0.0;
      for (uint32_t j = 0; j < size; ++j) {
        Ap[i] += matrix_[i][j] * p[j];
      }
    }

    // Compute alpha
    double pAp = dot(p, Ap);
    double global_pAp;
    boost::mpi::all_reduce(world, pAp, global_pAp, std::plus<double>());
    double alpha = rTr / global_pAp;

    // Update solution
    for (uint32_t i = 0; i < size; ++i) {
      result_[i] += alpha * p[i];
    }

    // Update residual
    for (uint32_t i = 0; i < size; ++i) {
      r[i] -= alpha * Ap[i];
    }

    // Check convergence
    double new_rTr = dot(r, r);
    double global_new_rTr;
    boost::mpi::all_reduce(world, new_rTr, global_new_rTr, std::plus<double>());
    if (std::sqrt(global_new_rTr) < tolerance) {
      break;
    }

    // Compute beta
    double beta = global_new_rTr / rTr;
    rTr = global_new_rTr;

    // Update search direction
    for (uint32_t i = 0; i < size; ++i) {
      p[i] = r[i] + beta * p[i];
    }
  }

  return true;
}

bool TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  }

  return true;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}
}  // namespace malyshev_v_conjugate_gradient_method