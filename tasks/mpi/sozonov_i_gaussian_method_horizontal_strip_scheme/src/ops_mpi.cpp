#include "mpi/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::extended_matrix_rank(int n, int m, std::vector<double> a) {
  const double EPS = 1e-9;

  int rank = m;
  for (int i = 0; i < m; ++i) {
    int j;
    for (j = 0; j < n; ++j) {
      if (std::abs(a[j * n + i]) > EPS) {
        break;
      }
    }
    if (j == n) {
      --rank;
    } else {
      for (int k = i + 1; k < m; ++k) {
        double ml = a[k * n + i] / a[i * n + i];
        for (j = i; j < n - 1; ++j) {
          a[k * n + j] -= a[i * n + j] * ml;
        }
      }
    }
  }
  return rank;
}

int sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::determinant(int n, int m, std::vector<double> a) {
  const double EPS = 1e-9;
  double det = 1;

  for (int i = 0; i < m; ++i) {
    int idx = i;
    for (int k = i + 1; k < m; ++k) {
      if (std::abs(a[k * n + i]) > std::abs(a[idx * n + i])) {
        idx = k;
      }
    }
    if (std::abs(a[idx * n + i]) < EPS) {
      return 0;
    }
    if (idx != i) {
      for (int j = 0; j < n - 1; ++j) {
        double tmp = a[i * n + j];
        a[i * n + j] = a[idx * n + j];
        a[idx * n + j] = tmp;
      }
      det *= -1;
    }
    det *= a[i * n + i];
    for (int k = i + 1; k < m; ++k) {
      double ml = a[k * n + i] / a[i * n + i];
      for (int j = i; j < n - 1; ++j) {
        a[k * n + j] -= a[i * n + j] * ml;
      }
    }
  }
  return det;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];
  // Init value for output
  x = std::vector<double>(cols - 1, 0);
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];

  // Check matrix for a single solution
  return taskData->inputs_count[0] > 0 && rows == cols - 1 && determinant(cols, rows, matrix) != 0 &&
         extended_matrix_rank(cols, rows, matrix) == rows;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows - 1; ++i) {
    for (int k = i + 1; k < rows; ++k) {
      double m = matrix[k * cols + i] / matrix[i * cols + i];
      for (int j = i; j < cols; ++j) {
        matrix[k * cols + j] -= matrix[i * cols + j] * m;
      }
    }
  }
  for (int i = rows - 1; i >= 0; --i) {
    double sum = matrix[i * cols + rows];
    for (int j = i + 1; j < cols - 1; ++j) {
      sum -= matrix[i * cols + j] * x[j];
    }
    x[i] = sum / matrix[i * cols + i];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cols - 1; ++i) {
    reinterpret_cast<double *>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init matrix
    matrix = std::vector<double>(taskData->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
    cols = taskData->inputs_count[1];
    rows = taskData->inputs_count[2];
    // Init value for output
    x = std::vector<double>(cols - 1, 0);
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init matrix
    matrix = std::vector<double>(taskData->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], matrix.begin());
    cols = taskData->inputs_count[1];
    rows = taskData->inputs_count[2];

    // Check matrix for a single solution
    return taskData->inputs_count[0] > 0 && rows == cols - 1 && determinant(cols, rows, matrix) != 0 &&
           extended_matrix_rank(cols, rows, matrix) == rows;
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, cols, 0);
  broadcast(world, rows, 0);

  std::vector<int> row_num(world.size());

  int delta = rows / world.size();
  if (rows % world.size() != 0) {
    delta++;
  }
  if (world.rank() >= world.size() - world.size() * delta + rows) {
    delta--;
  }

  boost::mpi::gather(world, delta, row_num.data(), 0);

  if (world.rank() == 0) {
    std::vector<double> send_matrix(delta * cols);
    for (int proc = 1; proc < world.size(); ++proc) {
      for (int i = 0; i < row_num[proc]; ++i) {
        for (int j = 0; j < cols; ++j) {
          send_matrix[i * cols + j] = matrix[(proc + world.size() * i) * cols + j];
        }
      }
      world.send(proc, 0, send_matrix.data(), row_num[proc] * cols);
    }
  }

  local_matrix = std::vector<double>(delta * cols);

  if (world.rank() == 0) {
    for (int i = 0; i < delta; ++i) {
      for (int j = 0; j < cols; ++j) {
        local_matrix[i * cols + j] = matrix[i * cols * world.size() + j];
      }
    }
  } else {
    world.recv(0, 0, local_matrix.data(), delta * cols);
  }

  std::vector<double> row(delta);
  for (int i = 0; i < delta; ++i) {
    row[i] = world.rank() + world.size() * i;
  }

  std::vector<double> pivot(cols);
  int r = 0;
  for (int i = 0; i < rows - 1; ++i) {
    if (i == row[r]) {
      for (int j = 0; j < cols; ++j) {
        pivot[j] = local_matrix[r * cols + j];
      }
      broadcast(world, pivot.data(), cols, world.rank());
      r++;
    } else {
      broadcast(world, pivot.data(), cols, i % world.size());
    }
    for (int k = r; k < delta; ++k) {
      double m = local_matrix[k * cols + i] / pivot[i];
      for (int j = i; j < cols; ++j) {
        local_matrix[k * cols + j] -= pivot[j] * m;
      }
    }
  }

  local_x = std::vector<double>(cols - 1, 0);
  r = 0;
  for (int i = 0; i < rows; ++i) {
    if (i == row[r]) {
      local_x[i] = local_matrix[r * cols + rows];
      r++;
    }
  }

  r = delta - 1;
  for (int i = rows - 1; i > 0; --i) {
    if (r >= 0) {
      if (i == row[r]) {
        local_x[i] /= local_matrix[r * cols + i];
        broadcast(world, local_x[i], world.rank());
        r--;
      } else {
        broadcast(world, local_x[i], i % world.size());
      }
    } else {
      broadcast(world, local_x[i], i % world.size());
    }
    if (r >= 0) {
      for (int j = 0; j <= r; ++j) {
        local_x[row[j]] -= local_matrix[j * cols + i] * local_x[i];
      }
    }
  }

  if (world.rank() == 0) {
    local_x[0] /= local_matrix[0];
    x = local_x;
  }

  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cols - 1; ++i) {
      reinterpret_cast<double *>(taskData->outputs[0])[i] = x[i];
    }
  }
  return true;
}