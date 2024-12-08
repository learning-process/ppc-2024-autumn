#include "mpi/gromov_a_gaussian_method_vertical/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

int gromov_a_gaussian_method_vertical_mpi::matrix_rank(std::vector<double>& matrix, int rows, int columns,
                                                       int band_width) {
  int rank = 0;

  for (int i = 0; i < rows; ++i) {
    double max_element = 0.0;
    int max_row = i;

    for (int k = i; k < std::min(i + band_width, rows); ++k) {
      if (std::abs(matrix[k * columns + i]) > max_element) {
        max_element = std::abs(matrix[k * columns + i]);
        max_row = k;
      }
    }

    if (max_element == 0) {
      continue;
    }

    for (int k = 0; k < columns; ++k) {
      std::swap(matrix[max_row * columns + k], matrix[i * columns + k]);
    }

    for (int k = i + 1; k < std::min(i + band_width, rows); ++k) {
      double factor = matrix[k * columns + i] / matrix[i * columns + i];
      for (int j = i; j < columns; ++j) {
        matrix[k * columns + j] -= factor * matrix[i * columns + j];
      }
    }
    rank++;
  }
  return rank;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential::pre_processing() {
  internal_order_test();
  equations = taskData->inputs_count[1];

  input_coefficient.assign(taskData->inputs_count[0], 0);
  std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
            reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_coefficient.begin());

  input_rhs.assign(taskData->inputs_count[1], 0);
  std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
            reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1], input_rhs.begin());

  res.resize(equations);

  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential::validation() {
  internal_order_test();
  std::vector<double> matrix_argument(equations * (equations + 1));

  for (int i = 0; i < equations; ++i) {
    for (int j = 0; j < equations; ++j) {
      matrix_argument[i * (equations + 1) + j] = static_cast<double>(input_coefficient[i * equations + j]);
    }
    matrix_argument[i * (equations + 1) + equations] = static_cast<double>(input_rhs[i]);
  }

  int rank_coeffs = matrix_rank(matrix_argument, equations, equations, band_width);
  int rank_augmented = matrix_rank(matrix_argument, equations, equations + 1, band_width);

  return (rank_coeffs == rank_augmented);
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential::run() {
  internal_order_test();
  std::vector<double> matrix_argument(equations * (equations + 1));

  for (int i = 0; i < equations; ++i) {
    for (int j = 0; j < equations; ++j) {
      matrix_argument[i * (equations + 1) + j] = static_cast<double>(input_coefficient[i * equations + j]);
    }
    matrix_argument[i * (equations + 1) + equations] = static_cast<double>(input_rhs[i]);
  }

  for (int i = 0; i < equations; ++i) {
    double max_element = std::abs(matrix_argument[i * (equations + 1) + i]);
    int max_row = i;

    for (int k = i + 1; k < std::min(i + band_width, equations); ++k) {
      if (std::abs(matrix_argument[k * (equations + 1) + i]) > max_element) {
        max_element = std::abs(matrix_argument[k * (equations + 1) + i]);
        max_row = k;
      }
    }
    for (int j = 0; j <= equations; ++j) {
      std::swap(matrix_argument[max_row * (equations + 1) + j], matrix_argument[i * (equations + 1) + j]);
    }

    for (int k = i + 1; k < equations; ++k) {
      double factor = matrix_argument[k * (equations + 1) + i] / matrix_argument[i * (equations + 1) + i];
      for (int j = i; j <= equations; ++j) {
        matrix_argument[k * (equations + 1) + j] -= factor * matrix_argument[i * (equations + 1) + j];
      }
    }
  }

  for (int i = equations - 1; i >= 0; --i) {
    res[i] = matrix_argument[i * (equations + 1) + equations];
    for (int j = i + 1; j < equations; ++j) {
      res[i] -= matrix_argument[i * (equations + 1) + j] * res[j];
    }
    res[i] /= matrix_argument[i * (equations + 1) + i];
  }
  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < equations; ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel::pre_processing() {
  internal_order_test();
  int proc_rank = world.rank();

  if (proc_rank == 0) {
    input_coefficient.assign(taskData->inputs_count[0], 0);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
              reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_coefficient.begin());

    input_rhs.assign(taskData->inputs_count[1], 0);
    std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
              reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1], input_rhs.begin());

    equations = taskData->inputs_count[1];
  }

  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->outputs_count[0] == 0 || taskData->inputs_count[0] == 0) return false;
    if (taskData->inputs_count[1] == 0) return false;

    int equations_valid = taskData->inputs_count[1];

    std::vector<double> validation_matrix(equations_valid * (equations_valid + 1));

    std::vector<double> input_coeff_valid(taskData->inputs_count[0]);
    std::transform(reinterpret_cast<int*>(taskData->inputs[0]),
                   reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_coeff_valid.begin(),
                   [](int coeff) { return static_cast<double>(coeff); });

    std::vector<int> input_rhs_valid(taskData->inputs_count[1]);
    std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
              reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1], input_rhs_valid.begin());

    for (int i = 0; i < equations_valid; ++i) {
      for (int j = 0; j < equations_valid; ++j) {
        validation_matrix[i * (equations_valid + 1) + j] = (input_coeff_valid[i * equations_valid + j]);
      }
      validation_matrix[i * (equations_valid + 1) + equations_valid] = static_cast<double>(input_rhs_valid[i]);
    }

    int rank_coeffs = matrix_rank(input_coeff_valid, equations_valid, equations_valid, band_width);
    int rank_augmented = matrix_rank(validation_matrix, equations_valid, equations_valid + 1, band_width);
    if (rank_coeffs != rank_augmented) {
      return false;
    }
  }
  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel::run() {
  internal_order_test();

  int proc_rank = world.rank();
  int proc_size = world.size();

  if (proc_rank == 0) {
    matrix_argument = std::vector<double>(equations * (equations + 1));
    changed_matrix = std::vector<double>(equations * (equations + 1));
    for (int i = 0; i < equations; ++i) {
      for (int j = 0; j < equations; ++j) {
        matrix_argument[i * (equations + 1) + j] = static_cast<double>(input_coefficient[i * equations + j]);
        changed_matrix[i * (equations + 1) + j] = static_cast<double>(input_coefficient[i * equations + j]);
      }
      matrix_argument[i * (equations + 1) + equations] = static_cast<double>(input_rhs[i]);
      changed_matrix[i * (equations + 1) + equations] = static_cast<double>(input_rhs[i]);
    }
    size_row = int(matrix_argument.size()) / equations;
    count_row_proc = equations / proc_size;
    remainder = equations % proc_size;
    count_row_proc += remainder;
  }

  broadcast(world, equations, 0);
  broadcast(world, count_row_proc, 0);
  broadcast(world, size_row, 0);
  res.resize(equations);

  if (proc_rank == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix_argument.data() + proc * count_row_proc * size_row, count_row_proc * size_row);
    }
  }
  local_matrix = std::vector<double>(count_row_proc * size_row);

  if (proc_rank == 0) {
    local_matrix = std::vector<double>(matrix_argument.begin(), matrix_argument.begin() + count_row_proc * size_row);
  } else {
    world.recv(0, 0, local_matrix.data(), count_row_proc * size_row);
  }
  local_max_row.resize(size_row);

  for (int i = 0; i < equations; ++i) {
    if (proc_rank == 0) {
      double max_elem = std::abs(changed_matrix[i * (equations + 1) + i]);
      int max_row = i;

      for (int k = 0; k < equations; ++k) {
        if (std::abs(changed_matrix[k * (equations + 1) + i]) > max_elem) {
          max_elem = std::abs(changed_matrix[k * (equations + 1) + i]);
          max_row = k;
        }
      }

      for (int j = 0; j < size_row; ++j) {
        local_max_row[j] = changed_matrix[max_row * size_row + j];
        res_matrix.push_back(local_max_row[j]);
      }

      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, local_max_row.data(), size_row);
      }
    }

    if (proc_rank != 0) {
      world.recv(0, 0, local_max_row.data(), size_row);
    }

    for (int k = 0; k < count_row_proc; k++) {
      double factor = local_matrix[k * size_row + i] / local_max_row[i];
      for (int j = i; j < size_row; j++) {
        local_matrix[k * size_row + j] -= factor * local_max_row[j];
      }
    }

    gather(world, local_matrix.data(), size_row * count_row_proc, changed_matrix, 0);
  }

  if (proc_rank == 0) {
    for (int i = equations - 1; i >= 0; --i) {
      res[i] = res_matrix[i * (equations + 1) + equations];
      for (int j = i + 1; j < equations; ++j) {
        res[i] -= res_matrix[i * (equations + 1) + j] * res[j];
      }
      res[i] /= res_matrix[i * (equations + 1) + i];
    }
  }
  return true;
}

bool gromov_a_gaussian_method_vertical_mpi::MPIGaussVerticalParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < equations; ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}