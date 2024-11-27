#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <iostream>
#include <vector>

namespace petrov_o_horizontal_gauss_method_mpi {

bool ParallelTask::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
      return false;
    }

    size_t n = taskData->inputs_count[0];

    if (n == 0) {
      return false;
    }

    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
      return false;
    }
  }

  return true;
}

bool ParallelTask::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_t n = taskData->inputs_count[0];

    matrix.resize(n, std::vector<double>(n));
    b.resize(n);
    x.resize(n);

    double* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        matrix[i][j] = matrix_input[i * n + j];
      }
    }

    double* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
    for (size_t i = 0; i < n; ++i) {
      b[i] = b_input[i];
    }
  }

  return true;
}

bool ParallelTask::run() {
  internal_order_test();

  size_t n = matrix.size();

  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, matrix, 0);
  boost::mpi::broadcast(world, b, 0);

  for (size_t k = 0; k < n - 1; ++k) {
    for (size_t i = k + 1 + world.rank(); i < n; i += world.size()) {
      double factor = matrix[i][k] / matrix[k][k];
      for (size_t j = k; j < n; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
      b[i] -= factor * b[k];
    }

    for (size_t i = k + 1; i < n; ++i) {
      boost::mpi::broadcast(world, matrix[i], (i - (k + 1)) % world.size());
      boost::mpi::broadcast(world, b[i], (i - (k + 1)) % world.size());
    }
  }

  if (world.rank() == 0) {
    x[n - 1] = b[n - 1] / matrix[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
      double sum = b[i];
      for (size_t j = i + 1; j < n; ++j) {
        sum -= matrix[i][j] * x[j];
      }
      x[i] = sum / matrix[i][i];
    }
  }

  return true;
}

bool ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    double* output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < x.size(); ++i) {
      output[i] = x[i];
    }
  }
  return true;
}

bool SequentialTask::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
    return false;
  }

  size_t n = taskData->inputs_count[0];

  if (n == 0) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  return true;
}

bool SequentialTask::pre_processing() {
  internal_order_test();

  size_t n = taskData->inputs_count[0];

  matrix.resize(n, std::vector<double>(n));
  b.resize(n);
  x.resize(n);

  double* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      matrix[i][j] = matrix_input[i * n + j];
    }
  }

  double* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
  for (size_t i = 0; i < n; ++i) {
    b[i] = b_input[i];
  }

  return true;
}

bool SequentialTask::run() {
  internal_order_test();

  size_t n = matrix.size();

  for (size_t k = 0; k < n - 1; ++k) {
    for (size_t i = k + 1; i < n; ++i) {
      double factor = matrix[i][k] / matrix[k][k];
      for (size_t j = k; j < n; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
      b[i] -= factor * b[k];
    }
  }

  x[n - 1] = b[n - 1] / matrix[n - 1][n - 1];
  for (int i = n - 2; i >= 0; --i) {
    double sum = b[i];
    for (size_t j = i + 1; j < n; ++j) {
      sum -= matrix[i][j] * x[j];
    }
    x[i] = sum / matrix[i][i];
  }

  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();

  double* output = reinterpret_cast<double*>(taskData->outputs[0]);
  for (size_t i = 0; i < x.size(); ++i) {
    output[i] = x[i];
  }
  return true;
}

}  // namespace petrov_o_horizontal_gauss_method_mpi
