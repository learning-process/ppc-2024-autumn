#include "mpi/kolokolova_d_gaussian_method_horizontal/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kolokolova_d_gaussian_method_horizontal_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  std::uniform_int_distribution<int> dist(1, 100);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

int kolokolova_d_gaussian_method_horizontal_mpi::find_rank(std::vector<double>& matrix, int rows, int cols) {
  int rank = 0;

  for (int i = 0; i < rows; ++i) {
    // Find max element
    double max_elem = 0.0;
    int max_row = i;
    for (int k = i; k < rows; ++k) {
      if (std::abs(matrix[k * cols + i]) > max_elem) {
        max_elem = std::abs(matrix[k * cols + i]);
        max_row = k;
      }
    }

    // If all matrice is 0, than rank = 0
    if (max_elem == 0) {
      continue;
    }

    // Rearranging rows to move the max element to the current position
    for (int k = 0; k < cols; ++k) {
      std::swap(matrix[max_row * cols + k], matrix[i * cols + k]);
    }

    // Make all elements below the current to zero
    for (int k = i + 1; k < rows; ++k) {
      double factor = matrix[k * cols + i] / matrix[i * cols + i];
      for (int j = i; j < cols; ++j) {
        matrix[k * cols + j] -= factor * matrix[i * cols + j];
      }
    }

    rank++;
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
  std::vector<double> matrix_argum(count_equations * (count_equations + 1));

  // Filling the matrix
  for (int i = 0; i < count_equations; ++i) {
    for (int j = 0; j < count_equations; ++j) {
      matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
    }
    matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
  }

  // Get rangs of matrices
  int rank_A = find_rank(matrix_argum, count_equations, count_equations);
  int rank_Ab = find_rank(matrix_argum, count_equations, count_equations + 1);

  // Checking for inconsistency
  if (rank_A != rank_Ab) {
    return false;
  }
  return true;
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
      //std::cout << "input [" << i << "] = " << input_coeff[i] << "\n";
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

    // Check that that the system has a solution
    std::vector<double> validation_matrix(count_equations * (count_equations + 1));

    // Filling the matrix
    for (int i = 0; i < count_equations; ++i) {
      for (int j = 0; j < count_equations; ++j) {
        validation_matrix[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
      }
      validation_matrix[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
    }

    // Get rangs of matrices
    int rank_A = find_rank(validation_matrix, count_equations, count_equations);
    int rank_Ab = find_rank(validation_matrix, count_equations, count_equations + 1);

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
    //for (int i = 0; i < int(matrix_argum.size()); i++) {
    //  std::cout << "[" << i << "] = " << matrix_argum[i] << "\n";
    //}
    size_row = int(matrix_argum.size()) / count_equations;
    count_row_proc = count_equations / proc_size;
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
  //std::cout << count_row_proc << " - count row in rang: " << proc_rank << "\n";
  //std::cout << size_row << " - size row in rang: " << proc_rank << "\n";
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
        res_matrix.push_back(local_max_row[j]); // res_matrix contain of all max_row
        //std::cout << "Send local max row " << j << " " << local_max_row[j] << "\n";
      }

      //for (int j = 0; j <= count_equations; ++j) {
      //  std::swap(changed_matrix[max_row * (count_equations + 1) + j], changed_matrix[i * (count_equations + 1) + j]);
      //}

      // Send for each proc max_row
      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, local_max_row.data(), size_row);
        //std::cout << "Send proc: " << proc_rank << "\n";
      }
    }

    // Recv every proc
    if (proc_rank != 0) {
      world.recv(0, 0, local_max_row.data(), size_row);
      //std::cout << "Recv proc: " << proc_rank << "\n";
    }

    for (int k = 0; k < count_row_proc; k++) {
      double factor = local_matrix[k * size_row + i] / local_max_row[i];
      for (int j = i; j < size_row; j++) {
        local_matrix[k * size_row + j] -= factor * local_max_row[j];
        //std::cout << "Local matrix " << j << " " << local_matrix[k * size_row + j] << "\n";
      }
    }

    gather(world, local_matrix.data(), size_row * count_row_proc, changed_matrix, 0);
  }

  //if (proc_rank == 0) {
  //  for (int i = 0; i < int(res_matrix.size()); i++) {
  //    std::cout << "Result matrix: " << res_matrix[i] << "\n";
  //  }
  //}

  if (proc_rank == 0) {
    // Gaussian reversal
    for (int i = count_equations - 1; i >= 0; --i) {
      res[i] = res_matrix[i * (count_equations + 1) + count_equations];
      for (int j = i + 1; j < count_equations; ++j) {
        res[i] -= res_matrix[i * (count_equations + 1) + j] * res[j];
      }
      res[i] /= res_matrix[i * (count_equations + 1) + i];
    }

    //for (int i = 0; i < int(res.size()); i++) {
    //  std::cout << "Result " << res[i] << "\n";
    //}
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