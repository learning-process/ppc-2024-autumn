#include "mpi/kavtorev_d_jacobi_method/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool kavtorev_d_jacobi_method_mpi::JacobiMethodSequentialTask::pre_processing() {
  internal_order_test();
  matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  matrix_A.assign(matrix_size * matrix_size, 0.0);
  vector_b.assign(matrix_size, 0.0);
  solution_x.assign(matrix_size, 1.0);

  auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* b_input = reinterpret_cast<double*>(taskData->inputs[2]);

  std::copy(A_input, A_input + matrix_size * matrix_size, matrix_A.begin());
  std::copy(b_input, b_input + matrix_size, vector_b.begin());

  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodSequentialTask::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    return false;
  }

  matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  if (matrix_size <= 0) {
    return false;
  }

  auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
  std::vector<double> A_vec(A_input, A_input + matrix_size * matrix_size);

  for (size_t i = 0; i < matrix_size; ++i) {
    double diag = std::fabs(A_vec[i * matrix_size + i]);
    double sum = 0.0;

    for (size_t j = 0; j < matrix_size; ++j) {
      if (i != j) {
        sum += std::fabs(A_vec[i * matrix_size + j]);
      }
    }

    if (diag <= sum) {
      return false;
    }

    if (diag == 0.0) {
      return false;
    }
  }

  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodSequentialTask::run() {
  internal_order_test();
  std::vector<double> x_prev(matrix_size, 1.0);

  size_t numberOfIter = 0;

  while (numberOfIter < maxIterations) {
    std::copy(solution_x.begin(), solution_x.end(), x_prev.begin());

    for (size_t k = 0; k < matrix_size; k++) {
      double S = 0;
      for (size_t j = 0; j < matrix_size; j++) {
        if (j != k) {
          S += matrix_A[k * matrix_size + j] * x_prev[j];
        }
      }
      solution_x[k] = (vector_b[k] - S) / matrix_A[k * matrix_size + k];
    }

    if (has_converged(x_prev, solution_x)) break;
    numberOfIter++;
  }

  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodSequentialTask::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < solution_x.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = solution_x[i];
  }
  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::pre_processing() {
  internal_order_test();
  sizes_a.resize(world.size());
  displs_a.resize(world.size());

  sizes_b.resize(world.size());
  displs_b.resize(world.size());

  if (world.rank() == 0) {
    matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    matrix_A.assign(matrix_size * matrix_size, 0.0);
    vector_b.assign(matrix_size, 0.0);
    solution_x.assign(matrix_size, 0.0);
    x_prev.assign(matrix_size, 0.0);

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* b_input = reinterpret_cast<double*>(taskData->inputs[2]);

    std::copy(A_input, A_input + matrix_size * matrix_size, matrix_A.begin());
    std::copy(b_input, b_input + matrix_size, vector_b.begin());

    calculate_distribution(matrix_size, world.size(), sizes_a, displs_a, matrix_size);
    calculate_distribution(matrix_size, world.size(), sizes_b, displs_b, 1);
  }

  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      return false;
    }

    matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    if (matrix_size <= 0) {
      return false;
    }

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    std::vector<double> A_vec(A_input, A_input + matrix_size * matrix_size);

    if (is_singular(A_vec, matrix_size)) {
      std::cerr << "Error: Matrix determinant is zero." << std::endl;
      return false;
    }

    for (size_t i = 0; i < matrix_size; ++i) {
      double diag = std::fabs(A_vec[i * matrix_size + i]);
      double sum = 0.0;

      for (size_t j = 0; j < matrix_size; ++j) {
        if (i != j) {
          sum += std::fabs(A_vec[i * matrix_size + j]);
        }
      }

      if (diag <= sum) {
        return false;
      }

      if (diag == 0.0) {
        return false;
      }
    }
  }
  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::run() {
  internal_order_test();
  std::vector<double> local_x;

  boost::mpi::broadcast(world, sizes_a, 0);
  boost::mpi::broadcast(world, sizes_b, 0);
  boost::mpi::broadcast(world, displs_b, 0);
  boost::mpi::broadcast(world, matrix_size, 0);

  int loc_mat_size = sizes_a[world.rank()];
  int loc_vec_size = sizes_b[world.rank()];

  local_A.resize(loc_mat_size);
  local_b.resize(loc_vec_size);
  local_x.resize(sizes_b[world.rank()]);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, matrix_A.data(), sizes_a, displs_a, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, vector_b.data(), sizes_b, displs_b, local_b.data(), loc_vec_size, 0);
  } else {
    boost::mpi::scatterv(world, local_A.data(), loc_mat_size, 0);
    boost::mpi::scatterv(world, local_b.data(), loc_vec_size, 0);
  }

  for (size_t numberOfIter = 0; numberOfIter < maxIterations; numberOfIter++) {
    if (world.rank() == 0) {
      std::copy(solution_x.begin(), solution_x.end(), x_prev.begin());
    }
    boost::mpi::broadcast(world, x_prev, 0);

    for (int k = 0; k < sizes_b[world.rank()]; k++) {
      double S = 0;
      for (int j = 0; j < static_cast<int>(matrix_size); j++) {
        if (j != (displs_b[world.rank()] + k)) {
          S += local_A[k * matrix_size + j] * x_prev[j];
        }
      }
      local_x[k] = (local_b[k] - S) / local_A[k * matrix_size + displs_b[world.rank()] + k];
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, local_x.data(), sizes_b[world.rank()], solution_x.data(), sizes_b, displs_b, 0);
    } else {
      boost::mpi::gatherv(world, local_x.data(), sizes_b[world.rank()], 0);
    }
    bool need;
    if (world.rank() == 0) {
      need = has_converged(x_prev, solution_x);
    }
    boost::mpi::broadcast(world, need, 0);

    if (need) break;
  }

  return true;
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < solution_x.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = solution_x[i];
    }
  }
  return true;
}

void kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::calculate_distribution(int total_size, int num_procs,
                                                                                    std::vector<int>& sizes,
                                                                                    std::vector<int>& displs,
                                                                                    int block_size) {
  sizes.resize(num_procs, 0);
  displs.resize(num_procs, 0);

  if (num_procs > total_size) {
    for (int i = 0; i < total_size; ++i) {
      sizes[i] = block_size;
      displs[i] = i * block_size;
    }
  } else {
    int base_size = total_size / num_procs;
    int remainder = total_size % num_procs;

    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
      sizes[i] = (base_size + (remainder-- > 0 ? 1 : 0)) * block_size;
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

bool kavtorev_d_jacobi_method_mpi::JacobiMethodParallelTask::is_singular(const std::vector<double>& A,
                                                                         size_t matrix_size) {
  std::vector<double> matrix = A;

  for (size_t k = 0; k < matrix_size; ++k) {
    if (std::fabs(matrix[k * matrix_size + k]) < 1e-10 * std::max(1.0, std::fabs(matrix[k * matrix_size + k]))) {
      bool swapped = false;

      for (size_t i = k + 1; i < matrix_size; ++i) {
        if (std::fabs(matrix[i * matrix_size + k]) > 1e-10) {
          for (size_t j = 0; j < matrix_size; ++j) {
            std::swap(matrix[k * matrix_size + j], matrix[i * matrix_size + j]);
          }

          swapped = true;
          break;
        }
      }

      if (!swapped) {
        return true;
      }
    }

    for (size_t i = k + 1; i < matrix_size; ++i) {
      double factor = matrix[i * matrix_size + k] / matrix[k * matrix_size + k];

      for (size_t j = k; j < matrix_size; ++j) {
        matrix[i * matrix_size + j] -= factor * matrix[k * matrix_size + j];
      }
    }
  }

  return false;
}
