#include "mpi/korablev_v_jacobi_method/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool korablev_v_jacobi_method_mpi::JacobiMethodSequential::isNeedToComplete(const std::vector<double>& x_old,
                                                                            const std::vector<double>& x_new) const {
  double sum_up = 0;
  double sum_low = 0;
  for (size_t k = 0; k < x_old.size(); k++) {
    sum_up += (x_new[k] - x_old[k]) * (x_new[k] - x_old[k]);
    sum_low += x_new[k] * x_new[k];
  }
  return (sqrt(sum_up / sum_low) < epsilon_);
}

bool korablev_v_jacobi_method_mpi::JacobiMethodSequential::pre_processing() {
  internal_order_test();
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  A_.resize(n * n);
  b_.resize(n);
  x_.resize(n, 1.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      A_[i * n + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }

  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    std::cerr << "Error: Invalid number of inputs or outputs." << std::endl;
    return false;
  }

  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (n <= 0) {
    std::cerr << "Error: Matrix size must be positive." << std::endl;
    return false;
  }

  for (size_t i = 0; i < n; ++i) {
    double diag = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
      }
    }

    if (diag <= sum) {
      std::cerr << "Error: Matrix is not diagonally dominant at row " << i << "." << std::endl;
      return false;
    }

    if (diag == 0.0) {
      std::cerr << "Error: Zero element on the diagonal at row " << i << "." << std::endl;
      return false;
    }
  }

  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodSequential::run() {
  internal_order_test();
  size_t n = b_.size();
  std::vector<double> x_prev(n, 1.0);

  size_t numberOfIter = 0;

  while (numberOfIter < maxIterations_) {
    std::copy(x_.begin(), x_.end(), x_prev.begin());

    for (size_t k = 0; k < n; k++) {
      double S = 0;
      for (size_t j = 0; j < n; j++) {
        if (j != k) {
          S += A_[k * n + j] * x_prev[j];
        }
      }
      x_[k] = (b_[k] - S) / A_[k * n + k];
    }

    if (isNeedToComplete(x_prev, x_)) break;
    numberOfIter++;
  }

  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodParallel::isNeedToComplete(const std::vector<double>& x_old,
                                                                          const std::vector<double>& x_new) const {
  double sum_up = 0;
  double sum_low = 0;
  for (size_t k = 0; k < x_old.size(); k++) {
    sum_up += (x_new[k] - x_old[k]) * (x_new[k] - x_old[k]);
    sum_low += x_new[k] * x_new[k];
  }
  return (sqrt(sum_up / sum_low) < epsilon_);
}

bool korablev_v_jacobi_method_mpi::JacobiMethodParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    A_.resize(n * n);
    b_.resize(n);
    x_.resize(n, 0.0);
    x_prev.resize(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        A_[i * n + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
      }
      b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    }
  }
  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      std::cerr << "Error: Invalid number of inputs or outputs." << std::endl;
      return false;
    }

    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    if (n <= 0) {
      std::cerr << "Error: Matrix size must be positive." << std::endl;
      return false;
    }

    for (size_t i = 0; i < n; ++i) {
      double diag = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
      double sum = 0.0;

      for (size_t j = 0; j < n; ++j) {
        if (i != j) {
          sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
        }
      }
      if (diag <= sum) {
        std::cerr << "Error: Matrix is not diagonally dominant at row " << i << "." << std::endl;
        return false;
      }

      if (diag == 0.0) {
        std::cerr << "Error: Zero element on the diagonal at row " << i << "." << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodParallel::run() {
  internal_order_test();
  size_t n = b_.size();

  size_t numberOfIter = 0;

  std::vector<int> sizes_a(world.size());
  std::vector<int> displs_a(world.size());

  std::vector<int> sizes_b(world.size());
  std::vector<int> displs_b(world.size());

  if (world.rank() == 0) {
    calculate_distribution_a(n, world.size(), sizes_a, displs_a);
    calculate_distribution_b(n, world.size(), sizes_b, displs_b);
  }

  std::vector<double> local_x(sizes_b[world.rank()]);

  boost::mpi::broadcast(world, sizes_a, 0);
  boost::mpi::broadcast(world, sizes_b, 0);
  boost::mpi::broadcast(world, displs_a, 0);
  boost::mpi::broadcast(world, displs_b, 0);

  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, x_, 0);
  boost::mpi::broadcast(world, x_prev, 0);

  int loc_mat_size = sizes_a[world.rank()];
  int loc_vec_size = sizes_b[world.rank()];

  local_A.resize(loc_mat_size);
  local_b.resize(loc_vec_size);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, A_.data(), sizes_a, displs_a, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, b_.data(), sizes_b, displs_b, local_b.data(), loc_vec_size, 0);
  } else {
    boost::mpi::scatterv(world, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, local_b.data(), loc_vec_size, 0);
  }

  local_x.resize(sizes_b[world.rank()]);

  world.barrier();

  while (numberOfIter < maxIterations_) {
    if (world.rank() == 0) {
      std::copy(x_.begin(), x_.end(), x_prev.begin());
    }
    boost::mpi::broadcast(world, x_prev, 0);

    for (int k = 0; k < sizes_b[world.rank()]; k++) {
      double S = 0;
      for (int j = 0; j < static_cast<int>(n); j++) {
        if (j != (displs_b[world.rank()] + k)) {
          S += local_A[k * n + j] * x_prev[j];
        }
      }
      local_x[k] = (local_b[k] - S) / local_A[k * n + displs_b[world.rank()] + k];
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, local_x.data(), sizes_b[world.rank()], x_.data(), sizes_b, displs_b, 0);
    } else {
      boost::mpi::gatherv(world, local_x.data(), sizes_b[world.rank()], 0);
    }
    bool need;
    if (world.rank() == 0) {
      need = isNeedToComplete(x_prev, x_);
    }
    boost::mpi::broadcast(world, need, 0);

    if (need) break;

    numberOfIter++;
    world.barrier();
  }

  return true;
}

bool korablev_v_jacobi_method_mpi::JacobiMethodParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
    }
  }
  return true;
}

void korablev_v_jacobi_method_mpi::JacobiMethodParallel::calculate_distribution_a(int rows, int num_proc,
                                                                                  std::vector<int>& sizes,
                                                                                  std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = rows;
      displs[i] = i * rows;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * rows;
      } else {
        sizes[i] = a * rows;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

void korablev_v_jacobi_method_mpi::JacobiMethodParallel::calculate_distribution_b(int len, int num_proc,
                                                                                  std::vector<int>& sizes,
                                                                                  std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > len) {
    for (int i = 0; i < len; ++i) {
      sizes[i] = 1;
      displs[i] = i;
    }
  } else {
    int a = len / num_proc;
    int b = len % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1);
      } else {
        sizes[i] = a;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}