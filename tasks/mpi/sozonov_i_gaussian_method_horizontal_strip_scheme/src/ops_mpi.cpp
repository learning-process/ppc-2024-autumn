#include "mpi/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  matrix = std::vector<double>(taskData->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    matrix[i] = tmp_ptr[i];
  }
  n = taskData->inputs_count[1];
  // Init value for output
  x = std::vector<double>(n, 0);
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of input and output
  return taskData->inputs_count[0] == taskData->inputs_count[1] * (taskData->inputs_count[1] + 1) &&
         taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < n - 1; ++i) {
    for (size_t k = i + 1; k < n; ++k) {
      double m = matrix[k * (n + 1) + i] / matrix[i * (n + 1) + i];
      for (size_t j = i; j < (n + 1); ++j) {
        matrix[k * (n + 1) + j] -= matrix[i * (n + 1) + j] * m;
      }
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double b = matrix[i * (n + 1) + n];
    for (size_t j = i + 1; j < n; ++j) {
      b -= matrix[i * (n + 1) + j] * x[j];
    }
    x[i] = b / matrix[i * (n + 1) + i];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < n; ++i) {
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
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      matrix[i] = tmp_ptr[i];
    }
    n = taskData->inputs_count[1];

    // Init value for output
    x = std::vector<double>(n, 0);
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of input and output
    return taskData->inputs_count[0] == taskData->inputs_count[1] * (taskData->inputs_count[1] + 1) &&
           taskData->outputs_count[0] == taskData->inputs_count[1];
  }
  return true;
}

bool sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, n, 0);

  std::vector<int> row_num(world.size());

  int delta = n / world.size();
  if (n % world.size()) {
    delta++;
  }
  if (world.rank() >= world.size() - world.size() * delta + n) {
    delta--;
  }

  boost::mpi::gather(world, delta, row_num.data(), 0);

  if (world.rank() == 0) {
    std::vector<double> send_matrix(delta * (n + 1));
    for (int proc = 1; proc < world.size(); proc++) {
      for (size_t i = 0; i < row_num[proc]; ++i) {
        for (size_t j = 0; j < n + 1; ++j) {
          send_matrix[i * (n + 1) + j] = matrix[(proc + world.size() * i) * (n + 1) + j];
        }
      }
      world.send(proc, 0, send_matrix.data(), row_num[proc] * (n + 1));
    }
  }

  local_matrix = std::vector<double>(delta * (n + 1));

  if (world.rank() == 0) {
    for (size_t i = 0; i < delta; ++i) {
      for (size_t j = 0; j < n + 1; ++j) {
        local_matrix[i * (n + 1) + j] = matrix[i * (n + 1) * world.size() + j];
      }
    }
  } else {
    world.recv(0, 0, local_matrix.data(), delta * (n + 1));
  }

  std::vector<double> row(delta);
  for (size_t i = 0; i < delta; ++i) {
    row[i] = world.rank() + world.size() * i;
  }

  std::vector<double> pivot(n + 1);
  int r = 0;
  for (size_t i = 0; i < n - 1; ++i) {
    if (i == row[r]) {
      for (size_t j = 0; j < n + 1; ++j) {
        pivot[j] = local_matrix[r * (n + 1) + j];
      }
      broadcast(world, pivot.data(), n + 1, world.rank());
      r++;
    } else {
      broadcast(world, pivot.data(), n + 1, i % world.size());
    }
    for (size_t k = r; k < delta; k++) {
      double m = local_matrix[k * (n + 1) + i] / pivot[i];
      for (size_t j = i; j < n + 1; ++j) {
        local_matrix[k * (n + 1) + j] -= pivot[j] * m;
      }
    }
  }

  local_x = std::vector<double>(n, 0);
  r = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i == row[r]) {
      local_x[i] = local_matrix[r * (n + 1) + n];
      r++;
    }
  }

  r = delta - 1;
  for (size_t i = n - 1; i > 0; --i) {
    if (r >= 0) {
      if (i == row[r]) {
        local_x[i] /= local_matrix[r * (n + 1) + i];
        broadcast(world, local_x[i], world.rank());
        r--;
      } else {
        broadcast(world, local_x[i], i % world.size());
      }
    } else {
      broadcast(world, local_x[i], i % world.size());
    }
    if (r >= 0) {
      for (size_t j = 0; j <= r; ++j) {
        local_x[row[j]] -= local_matrix[j * (n + 1) + i] * local_x[i];
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
    for (size_t i = 0; i < n; ++i) {
      reinterpret_cast<double *>(taskData->outputs[0])[i] = x[i];
    }
  }
  return true;
}