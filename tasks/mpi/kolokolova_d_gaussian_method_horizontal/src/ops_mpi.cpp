#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

int kolokolova_d_gaussian_method_horizontal_mpi::find_rank(std::vector<double>& matrix, int rows, int cols) {
  std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat[i][j] = matrix[i * cols + j];
    }
  }

  int rank = 0;

  for (int col = 0; col < cols; ++col) {
    if (rank >= rows) break;
    int pivotRow = rank;
    while (pivotRow < rows && mat[pivotRow][col] == 0) {
      ++pivotRow;
    }
    if (pivotRow < rows) {
      for (int r = rank + 1; r < rows; ++r) {
        double factor = mat[r][col] / mat[rank][col];
        for (int j = col; j < cols; ++j) {
          mat[r][j] -= factor * mat[rank][j];
        }
      }
      ++rank;
    }
  }
  return rank;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  count_equations = taskData->inputs_count[1];

  // Init value for input and output
  input_coeff = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr_coeff = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_coeff[i] = tmp_ptr_coeff[i];
  }

  input_y = std::vector<int>(taskData->inputs_count[1]);
  auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
    input_y[i] = tmp_ptr_y[i];
  }
  res.resize(count_equations);
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  int count_equations_valid = taskData->inputs_count[1];

  // Check that that the system has a solution
  std::vector<double> validation_matrix(count_equations_valid * (count_equations_valid + 1));

  std::vector<double> input_coeff_valid(taskData->inputs_count[0]);
  auto* tmp_ptr_coeff = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_coeff_valid[i] = static_cast<double>(tmp_ptr_coeff[i]);
  }

  std::vector<int> input_y_valid(taskData->inputs_count[1]);
  auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
    input_y_valid[i] = tmp_ptr_y[i];
  }

  // Filling the matrix
  for (int i = 0; i < count_equations_valid; ++i) {
    for (int j = 0; j < count_equations_valid; ++j) {
      validation_matrix[i * (count_equations_valid + 1) + j] = (input_coeff_valid[i * count_equations_valid + j]);
    }
    validation_matrix[i * (count_equations_valid + 1) + count_equations_valid] = static_cast<double>(input_y_valid[i]);
  }

  // Get rangs of matrices
  int rank_A = find_rank(input_coeff_valid, count_equations_valid, count_equations_valid);
  int rank_Ab = find_rank(validation_matrix, count_equations_valid, count_equations_valid + 1);

  // Checking for inconsistency
  return (rank_A == rank_Ab);
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<double> matrix_argum(count_equations * (count_equations + 1));
  // Filling the matrix
  for (int i = 0; i < count_equations; ++i) {
    for (int j = 0; j < count_equations; ++j) {
      matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
    }
    matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
  }

  // Forward Gaussian move
  for (int i = 0; i < count_equations; ++i) {
    // Find max of element
    double max_elem = std::abs(matrix_argum[i * (count_equations + 1) + i]);
    int max_row = i;
    for (int k = i + 1; k < count_equations; ++k) {
      if (std::abs(matrix_argum[k * (count_equations + 1) + i]) > max_elem) {
        max_elem = std::abs(matrix_argum[k * (count_equations + 1) + i]);
        max_row = k;
      }
    }
    for (int j = 0; j <= count_equations; ++j) {
      std::swap(matrix_argum[max_row * (count_equations + 1) + j], matrix_argum[i * (count_equations + 1) + j]);
    }

    // Division by max element and subtraction
    for (int k = i + 1; k < count_equations; ++k) {
      double factor = matrix_argum[k * (count_equations + 1) + i] / matrix_argum[i * (count_equations + 1) + i];
      for (int j = i; j <= count_equations; ++j) {
        matrix_argum[k * (count_equations + 1) + j] -= factor * matrix_argum[i * (count_equations + 1) + j];
      }
    }
  }

  // Gaussian reversal
  for (int i = count_equations - 1; i >= 0; --i) {
    res[i] = matrix_argum[i * (count_equations + 1) + count_equations];
    for (int j = i + 1; j < count_equations; ++j) {
      res[i] -= matrix_argum[i * (count_equations + 1) + j] * res[j];
    }
    res[i] /= matrix_argum[i * (count_equations + 1) + i];
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_equations; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int proc_rank = world.rank();

  if (proc_rank == 0) {
    // Init value for input and output
    input_coeff = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr_coeff = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_coeff[i] = tmp_ptr_coeff[i];
    }

    input_y = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
    for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
      input_y[i] = tmp_ptr_y[i];
    }

    count_equations = taskData->inputs_count[1];
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output and input
    if (taskData->outputs_count[0] == 0 || taskData->inputs_count[0] == 0) return false;
    if (taskData->inputs_count[1] == 0) return false;

    int count_equations_valid = taskData->inputs_count[1];

    // Check that that the system has a solution
    std::vector<double> validation_matrix(count_equations_valid * (count_equations_valid + 1));

    std::vector<double> input_coeff_valid(taskData->inputs_count[0]);
    auto* tmp_ptr_coeff = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_coeff_valid[i] = static_cast<double>(tmp_ptr_coeff[i]);
    }

    std::vector<int> input_y_valid(taskData->inputs_count[1]);
    auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
    for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
      input_y_valid[i] = tmp_ptr_y[i];
    }

    // Filling the matrix
    for (int i = 0; i < count_equations_valid; ++i) {
      for (int j = 0; j < count_equations_valid; ++j) {
        validation_matrix[i * (count_equations_valid + 1) + j] = (input_coeff_valid[i * count_equations_valid + j]);
      }
      validation_matrix[i * (count_equations_valid + 1) + count_equations_valid] =
          static_cast<double>(input_y_valid[i]);
    }

    // Get rangs of matrices
    int rank_A = find_rank(input_coeff_valid, count_equations_valid, count_equations_valid);
    int rank_Ab = find_rank(validation_matrix, count_equations_valid, count_equations_valid + 1);
    // Checking for inconsistency
    if (rank_A != rank_Ab) {
      return false;
    }
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int proc_rank = world.rank();
  int proc_size = world.size();

  if (proc_rank == 0) {
    matrix_argum = std::vector<double>(count_equations * (count_equations + 1));
    changed_matrix = std::vector<double>(count_equations * (count_equations + 1));
    // Filling the matrix
    for (int i = 0; i < count_equations; ++i) {
      for (int j = 0; j < count_equations; ++j) {
        matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
        changed_matrix[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
      }
      matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
      changed_matrix[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
    }
    size_row = int(matrix_argum.size()) / count_equations;
    count_row_proc = count_equations / proc_size;
    remainder = count_equations % proc_size;
    count_row_proc += remainder;
  }

  // Send data to all proc
  broadcast(world, count_equations, 0);
  broadcast(world, count_row_proc, 0);
  broadcast(world, size_row, 0);
  res.resize(count_equations);

  if (proc_rank == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix_argum.data() + proc * count_row_proc * size_row, count_row_proc * size_row);
    }
  }
  local_matrix = std::vector<double>(count_row_proc * size_row);

  if (proc_rank == 0) {
    local_matrix = std::vector<double>(matrix_argum.begin(), matrix_argum.begin() + count_row_proc * size_row);
  } else {
    world.recv(0, 0, local_matrix.data(), count_row_proc * size_row);
  }
  local_max_row.resize(size_row);

  // Forward Gaussian move
  for (int i = 0; i < count_equations; ++i) {
    // Find max of element
    if (proc_rank == 0) {
      double max_elem = std::abs(changed_matrix[i * (count_equations + 1) + i]);
      int max_row = i;
      for (int k = 0; k < count_equations; ++k) {
        if (std::abs(changed_matrix[k * (count_equations + 1) + i]) > max_elem) {
          max_elem = std::abs(changed_matrix[k * (count_equations + 1) + i]);
          max_row = k;
        }
      }

      // Put elemnts in max_row
      for (int j = 0; j < size_row; ++j) {
        local_max_row[j] = changed_matrix[max_row * size_row + j];
        res_matrix.push_back(local_max_row[j]);  // res_matrix contain of all max_row
      }

      // Send for each proc max_row
      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, local_max_row.data(), size_row);
      }
    }

    // Recv every proc
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
    // Gaussian reversal
    for (int i = count_equations - 1; i >= 0; --i) {
      res[i] = res_matrix[i * (count_equations + 1) + count_equations];
      for (int j = i + 1; j < count_equations; ++j) {
        res[i] -= res_matrix[i * (count_equations + 1) + j] * res[j];
      }
      res[i] /= res_matrix[i * (count_equations + 1) + i];
    }
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < count_equations; i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}