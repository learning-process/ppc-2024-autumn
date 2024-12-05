#include "mpi/petrov_o_horizontal_gauss_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>

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

    auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
    std::vector<double> valid_matrix(matrix_input, matrix_input + n * n);

    int rank_matrix = 0;

    const double eps = 1e-10;

    for (size_t i = 0; i < n; ++i) {
      double max_elem = std::abs(valid_matrix[i * n + i]);
      size_t max_row = i;
      for (size_t k = i + 1; k < n; ++k) {
        if (std::abs(valid_matrix[k * n + i]) > max_elem) {
          max_elem = std::abs(valid_matrix[k * n + i]);
          max_row = k;
        }
      }

      if (max_elem < eps) {
        continue;
      }

      if (max_row != i) {
        for (size_t j = 0; j < n; ++j) {
          std::swap(valid_matrix[i * n + j], valid_matrix[max_row * n + j]);
        }
      }

      double pivot = valid_matrix[i * n + i];
      for (size_t j = i; j < n; ++j) {
        valid_matrix[i * n + j] /= pivot;
      }

      for (size_t k = i + 1; k < n; ++k) {
        double factor = valid_matrix[k * n + i];
        for (size_t j = i; j < n; ++j) {
          valid_matrix[k * n + j] -= factor * valid_matrix[i * n + j];
        }
      }

      ++rank_matrix;
    }

    if (rank_matrix != static_cast<int>(n)) {
      return false;
    }
  }
  return true;
}

bool ParallelTask::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_t n = taskData->inputs_count[0];

    matrix.resize(n * n);
    b.resize(n);
    x.resize(n);

    auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(matrix_input, matrix_input + n * n, matrix.begin());

    auto* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(b_input, b_input + n, b.begin());
  }

  return true;
}

bool ParallelTask::run() {
  internal_order_test();

  size_t n;

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
  }

  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, matrix, 0);
  boost::mpi::broadcast(world, b, 0);

  for (size_t k = 0; k < n - 1; ++k) {
    for (size_t i = k + 1 + world.rank(); i < n; i += world.size()) {
      double factor = matrix[i * n + k] / matrix[k * n + k];
      for (size_t j = k; j < n; ++j) {
        matrix[i * n + j] -= factor * matrix[k * n + j];
      }
      b[i] -= factor * b[k];
    }

    for (size_t i = k + 1; i < n; ++i) {
      std::vector<double> row(n);
      if (world.rank() == static_cast<int>(i - (k + 1)) % world.size()) {
        for (size_t j = 0; j < n; ++j) row[j] = matrix[i * n + j];
      }
      boost::mpi::broadcast(world, row, (i - (k + 1)) % world.size());

      for (size_t j = 0; j < n; ++j) matrix[i * n + j] = row[j];
      boost::mpi::broadcast(world, b[i], (i - (k + 1)) % world.size());
    }
  }

  if (world.rank() == 0) {
    x[n - 1] = b[n - 1] / matrix[(n - 1) * n + n - 1];
    for (int i = n - 2; i >= 0; --i) {
      double sum = b[i];
      for (size_t j = i + 1; j < n; ++j) {
        sum -= matrix[i * n + j] * x[j];
      }
      x[i] = sum / matrix[i * n + i];
    }
  }

  return true;
}

bool ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(x.begin(), x.end(), output);
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

  auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
  std::vector<double> valid_matrix(matrix_input, matrix_input + n * n);

  int rank_matrix = 0;

  const double eps = 1e-10;

  for (size_t i = 0; i < n; ++i) {
    double max_elem = std::abs(valid_matrix[i * n + i]);
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(valid_matrix[k * n + i]) > max_elem) {
        max_elem = std::abs(valid_matrix[k * n + i]);
        max_row = k;
      }
    }

    if (max_elem < eps) {
      continue;
    }

    if (max_row != i) {
      for (size_t j = 0; j < n; ++j) {
        std::swap(valid_matrix[i * n + j], valid_matrix[max_row * n + j]);
      }
    }

    double pivot = valid_matrix[i * n + i];
    for (size_t j = i; j < n; ++j) {
      valid_matrix[i * n + j] /= pivot;
    }

    for (size_t k = i + 1; k < n; ++k) {
      double factor = valid_matrix[k * n + i];
      for (size_t j = i; j < n; ++j) {
        valid_matrix[k * n + j] -= factor * valid_matrix[i * n + j];
      }
    }

    ++rank_matrix;
  }

  return rank_matrix == static_cast<int>(n);
}

bool SequentialTask::pre_processing() {
  internal_order_test();

  size_t n = taskData->inputs_count[0];

  matrix.resize(n * n);
  b.resize(n);
  x.resize(n);

  auto* matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(matrix_input, matrix_input + n * n, matrix.begin());

  auto* b_input = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(b_input, b_input + n, b.begin());

  return true;
}

bool SequentialTask::run() {
  internal_order_test();

  size_t n = taskData->inputs_count[0];

  for (size_t k = 0; k < n - 1; ++k) {
    for (size_t i = k + 1; i < n; ++i) {
      double factor = matrix[i * n + k] / matrix[k * n + k];
      for (size_t j = k; j < n; ++j) {
        matrix[i * n + j] -= factor * matrix[k * n + j];
      }
      b[i] -= factor * b[k];
    }
  }

  x[n - 1] = b[n - 1] / matrix[(n - 1) * n + (n - 1)];
  for (int i = n - 2; i >= 0; --i) {
    double sum = b[i];
    for (size_t j = i + 1; j < n; ++j) {
      sum -= matrix[i * n + j] * x[j];
    }
    x[i] = sum / matrix[i * n + i];
  }

  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();

  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x.begin(), x.end(), output);
  return true;
}

}  // namespace petrov_o_horizontal_gauss_method_mpi
