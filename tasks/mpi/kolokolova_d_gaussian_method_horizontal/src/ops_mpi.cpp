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
  std::uniform_int_distribution<int> dist(-100, 99);
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
    }

    input_y = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
    for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
      input_y[i] = tmp_ptr_y[i];
    }

    count_equations = taskData->inputs_count[1];
  }
  res.resize(count_equations);

  //int proc_rank = world.rank();

  //if (proc_rank == 0) {
  //  delta = taskData->inputs_count[0] / world.size();
  //}

  //if (proc_rank == 0) {
  //  // Init vectors
  //  input_ = std::vector<int>(taskData->inputs_count[0]);
  //  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  //  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
  //    input_[i] = tmp_ptr[i];
  //  }
  //}
  //// Init value for output
  //res.resize(world.size());
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  //if (world.rank() == 0) {
  //  // Check count elements of output and input
  //  if (taskData->outputs_count[0] == 0 || taskData->inputs_count[0] == 0) return false;
  //}
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int proc_rank = world.rank();
  int proc_size = world.size();

  if (proc_rank == 0) {
    matrix_argum = std::vector<double>(count_equations * (count_equations + 1));
    // Filling the matrix
    for (int i = 0; i < count_equations; ++i) {
      for (int j = 0; j < count_equations; ++j) {
        matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
        std::cout << matrix_argum[i] << "\n";
      }
      matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
      //std::cout << matrix_argum[i] << "\n";
    }

    size_row = int(matrix_argum.size()) / count_equations;
    count_row_proc = count_equations / proc_size;
  }
  broadcast(world, count_equations, 0);
  broadcast(world, count_row_proc, 0);
  broadcast(world, size_row, 0);

  if (proc_rank == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix_argum.data() + proc * count_row_proc * size_row, count_row_proc * size_row);
    }
  }
  std::cout << count_row_proc << " - count row in rang: " << proc_rank << "\n";
  std::cout << size_row << " - size row in rang: " << proc_rank << "\n";
  local_matrix = std::vector<double>(count_row_proc * size_row);

  if (proc_rank == 0) {
    local_matrix = std::vector<double>(matrix_argum.begin(), matrix_argum.begin() + count_row_proc * size_row);
  } else {
    world.recv(0, 0, local_matrix.data(), count_row_proc * size_row);
    std::cout << count_row_proc << " - count row in rang: " << proc_rank << "\n";
    std::cout << size_row << " - size row in rang: " << proc_rank << "\n";
  }




  //double max_elem = 0;
  //double max_local_elem = 0;
  //int max_proc_row = 0;
  //// Forward Gaussian move
  //for (int i = 0; i < count_row_proc; ++i) {
  //  // Find local max of element
  //  max_local_elem = std::abs(local_matrix[i * (count_row_proc + 1) + i]);
  //  max_proc_row = i;
  //  for (int k = i + 1; k < count_row_proc; ++k) {
  //    if (std::abs(local_matrix[k * (count_row_proc + 1) + i]) > max_local_elem) {
  //      max_local_elem = std::abs(local_matrix[k * (count_row_proc + 1) + i]);
  //      max_proc_row = k;
  //    }
  //  }
  //  for (int j = 0; j <= count_row_proc; ++j) {
  //    std::swap(local_matrix[max_proc_row * (count_row_proc + 1) + j], local_matrix[i * (count_row_proc + 1) + j]);
  //  }
  //}
  //reduce(world, max_local_elem, max_elem, boost::mpi::maximum<double>(), 0);
  //broadcast(world, max_elem, 0);



 /* int proc_rank = world.rank();

  broadcast(world, delta, 0);

  if (proc_rank == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  local_input_ = std::vector<int>(delta);

  if (proc_rank == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  int local_res = 0;
  for (int i = 0; i < int(local_input_.size()); i++) {
    if (local_res < local_input_[i]) local_res = local_input_[i];
  }
  gather(world, local_res, res, 0);*/
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