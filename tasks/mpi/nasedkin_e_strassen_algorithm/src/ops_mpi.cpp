#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

namespace nasedkin_e_strassen_algorithm_mpi {

bool StrassenAlgorithmSequential::pre_processing() {
  internal_order_test();
  size_t matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  A_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));
  B_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));
  C_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));

  auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
  auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      A_[i][j] = A_input[i * matrix_size + j];
      B_[i][j] = B_input[i * matrix_size + j];
    }
  }
  std::cout << "seq preporcessing done" << std::endl;
  return true;
}

bool StrassenAlgorithmSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
    return false;
  }

  n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return n > 0;
  std::cout << "seq validation done" << std::endl;
}

bool StrassenAlgorithmSequential::run() {
  internal_order_test();
  strassen(A_, B_, C_);
  return true;
  std::cout << "seq run done" << std::endl;
}

bool StrassenAlgorithmSequential::post_processing() {
  internal_order_test();
  auto* C_output = reinterpret_cast<double*>(taskData->outputs[0]);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C_output[i * n + j] = C_[i][j];
    }
  }
  return true;
  std::cout << "seq postprocessing done" << std::endl;
}

void StrassenAlgorithmSequential::strassen(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  if (n == 1) {
    C[0][0] = A[0][0] * B[0][0];
    return;
  }

  size_t half = n / 2;

  std::vector<std::vector<double>> A11(half, std::vector<double>(half));
  std::vector<std::vector<double>> A12(half, std::vector<double>(half));
  std::vector<std::vector<double>> A21(half, std::vector<double>(half));
  std::vector<std::vector<double>> A22(half, std::vector<double>(half));

  std::vector<std::vector<double>> B11(half, std::vector<double>(half));
  std::vector<std::vector<double>> B12(half, std::vector<double>(half));
  std::vector<std::vector<double>> B21(half, std::vector<double>(half));
  std::vector<std::vector<double>> B22(half, std::vector<double>(half));

  std::vector<std::vector<double>> C11(half, std::vector<double>(half));
  std::vector<std::vector<double>> C12(half, std::vector<double>(half));
  std::vector<std::vector<double>> C21(half, std::vector<double>(half));
  std::vector<std::vector<double>> C22(half, std::vector<double>(half));

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + half];
      A21[i][j] = A[i + half][j];
      A22[i][j] = A[i + half][j + half];

      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + half];
      B21[i][j] = B[i + half][j];
      B22[i][j] = B[i + half][j + half];
    }
  }

  std::vector<std::vector<double>> M1;
  std::vector<std::vector<double>> M2;
  std::vector<std::vector<double>> M3;
  std::vector<std::vector<double>> M4;
  std::vector<std::vector<double>> M5;
  std::vector<std::vector<double>> M6;
  std::vector<std::vector<double>> M7;

  add(A11, A22, M1);
  add(B11, B22, M2);
  strassen(M1, M2, M3);

  add(A21, A22, M1);
  strassen(M1, B11, M4);

  subtract(B12, B22, M1);
  strassen(A11, M1, M5);

  subtract(B21, B11, M1);
  strassen(A22, M1, M6);

  add(A11, A12, M1);
  strassen(M1, B22, M7);

  add(M3, M5, C12);
  add(M2, M4, C21);

  add(M1, M3, M2);
  subtract(M2, M7, C11);

  add(M1, M6, M2);
  subtract(M2, M4, C22);

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      C[i][j] = C11[i][j];
      C[i][j + half] = C12[i][j];
      C[i + half][j] = C21[i][j];
      C[i + half][j + half] = C22[i][j];
    }
  }
}

void StrassenAlgorithmSequential::add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void StrassenAlgorithmSequential::subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
}

bool StrassenAlgorithmParallel::pre_processing() {
  internal_order_test();
  sizes_a.resize(world.size());
  displs_a.resize(world.size());

  sizes_b.resize(world.size());
  displs_b.resize(world.size());

  if (world.rank() == 0) {
    size_t matrix_size = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    A_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));
    B_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));
    C_.assign(matrix_size, std::vector<double>(matrix_size, 0.0));

    auto* A_input = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* B_input = reinterpret_cast<double*>(taskData->inputs[2]);

    for (size_t i = 0; i < matrix_size; ++i) {
      for (size_t j = 0; j < matrix_size; ++j) {
        A_[i][j] = A_input[i * matrix_size + j];
        B_[i][j] = B_input[i * matrix_size + j];
      }
    }

    calculate_distribution(matrix_size, matrix_size, world.size(), sizes_a, displs_a);
    calculate_distribution(matrix_size, matrix_size, world.size(), sizes_b, displs_b);
  }
  std::cout << "mpi preporcessing done" << std::endl;
  return true;
}

bool StrassenAlgorithmParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.size() != 3 || taskData->outputs_count.size() != 1) {
      return false;
    }

    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    return n > 0;
  }
  std::cout << "mpi validation done" << std::endl;
  return true;
}

bool StrassenAlgorithmParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, sizes_a, 0);
  boost::mpi::broadcast(world, sizes_b, 0);
  boost::mpi::broadcast(world, displs_a, 0);
  boost::mpi::broadcast(world, displs_b, 0);
  boost::mpi::broadcast(world, n, 0);

  int loc_mat_size = sizes_a[world.rank()];

  std::vector<double> local_A_flat(loc_mat_size * n);
  std::vector<double> local_B_flat(loc_mat_size * n);
  std::vector<double> local_C_flat(loc_mat_size * n);

  if (world.rank() == 0) {
    auto A_flat = flatten_matrix(A_);
    auto B_flat = flatten_matrix(B_);

    boost::mpi::scatterv(world, A_flat, sizes_a, displs_a, local_A_flat.data(), sizes_a[0], 0);
    boost::mpi::scatterv(world, B_flat, sizes_b, displs_b, local_B_flat.data(), sizes_b[0], 0);
  } else {
    boost::mpi::scatterv(world, local_A_flat.data(), sizes_a[world.rank()], 0);
    boost::mpi::scatterv(world, local_B_flat.data(), sizes_b[world.rank()], 0);
  }

  local_A = unflatten_matrix(local_A_flat, loc_mat_size, n);
  local_B = unflatten_matrix(local_B_flat, loc_mat_size, n);

  strassen_mpi(local_A, local_B, local_C);

  local_C_flat = flatten_matrix(local_C);

  if (world.rank() == 0) {
    std::vector<double> C_flat(n * n);
    boost::mpi::gatherv(world, local_C_flat.data(), sizes_a[world.rank()], C_flat.data(), sizes_a, displs_a, 0);
    C_ = unflatten_matrix(C_flat, n, n);
  } else {
    boost::mpi::gatherv(world, local_C_flat.data(), sizes_a[world.rank()], 0);
  }
  std::cout << "mpi run done" << std::endl;
  return true;
}

bool StrassenAlgorithmParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* C_output = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        C_output[i * n + j] = C_[i][j];
      }
    }
  }
  return true;
  std::cout << "mpi postprocessing done" << std::endl;
}

void StrassenAlgorithmParallel::calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = rows * cols;
      displs[i] = i * rows * cols;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * cols;
      } else {
        sizes[i] = a * cols;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

void StrassenAlgorithmParallel::distribute_matrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& local_matrix, const std::vector<int>& sizes, const std::vector<int>& displs) {
  for (int i = 0; i < world.size(); ++i) {
    int start = displs[i] / matrix.size();
    int end = start + sizes[i] / matrix.size();
    for (int j = start; j < end; ++j) {
      for (int k = 0; k < static_cast<int>(matrix.size()); ++k) {
        local_matrix[j - start][k] = matrix[j][k];
      }
    }
  }
}

void StrassenAlgorithmParallel::gather_matrix(std::vector<std::vector<double>>& matrix, const std::vector<std::vector<double>>& local_matrix, const std::vector<int>& sizes, const std::vector<int>& displs) {
  for (int i = 0; i < world.size(); ++i) {
    int start = displs[i] / matrix.size();
    int end = start + sizes[i] / matrix.size();
    for (int j = start; j < end; ++j) {
      for (int k = 0; k < static_cast<int>(matrix.size()); ++k) {
        matrix[j][k] = local_matrix[j - start][k];
      }
    }
  }
}

void StrassenAlgorithmParallel::strassen_mpi(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  if (n == 1) {
    C[0][0] = A[0][0] * B[0][0];
    return;
  }

  size_t half = n / 2;

  std::vector<std::vector<double>> A11(half, std::vector<double>(half));
  std::vector<std::vector<double>> A12(half, std::vector<double>(half));
  std::vector<std::vector<double>> A21(half, std::vector<double>(half));
  std::vector<std::vector<double>> A22(half, std::vector<double>(half));

  std::vector<std::vector<double>> B11(half, std::vector<double>(half));
  std::vector<std::vector<double>> B12(half, std::vector<double>(half));
  std::vector<std::vector<double>> B21(half, std::vector<double>(half));
  std::vector<std::vector<double>> B22(half, std::vector<double>(half));

  std::vector<std::vector<double>> C11(half, std::vector<double>(half));
  std::vector<std::vector<double>> C12(half, std::vector<double>(half));
  std::vector<std::vector<double>> C21(half, std::vector<double>(half));
  std::vector<std::vector<double>> C22(half, std::vector<double>(half));

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + half];
      A21[i][j] = A[i + half][j];
      A22[i][j] = A[i + half][j + half];

      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + half];
      B21[i][j] = B[i + half][j];
      B22[i][j] = B[i + half][j + half];
    }
  }

  std::vector<std::vector<double>> M1;
  std::vector<std::vector<double>> M2;
  std::vector<std::vector<double>> M3;
  std::vector<std::vector<double>> M4;
  std::vector<std::vector<double>> M5;
  std::vector<std::vector<double>> M6;
  std::vector<std::vector<double>> M7;

  add(A11, A22, M1);
  add(B11, B22, M2);
  strassen_mpi(M1, M2, M3);

  add(A21, A22, M1);
  strassen_mpi(M1, B11, M4);

  subtract(B12, B22, M1);
  strassen_mpi(A11, M1, M5);

  subtract(B21, B11, M1);
  strassen_mpi(A22, M1, M6);

  add(A11, A12, M1);
  strassen_mpi(M1, B22, M7);

  add(M3, M5, C12);
  add(M2, M4, C21);

  add(M1, M3, M2);
  subtract(M2, M7, C11);

  add(M1, M6, M2);
  subtract(M2, M4, C22);

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      C[i][j] = C11[i][j];
      C[i][j + half] = C12[i][j];
      C[i + half][j] = C21[i][j];
      C[i + half][j + half] = C22[i][j];
    }
  }
}

void StrassenAlgorithmParallel::add(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void StrassenAlgorithmParallel::subtract(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  size_t n = A.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
}

std::vector<double> StrassenAlgorithmParallel::flatten_matrix(const std::vector<std::vector<double>>& matrix) {
  std::vector<double> flat;
  for (const auto& row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

std::vector<std::vector<double>> StrassenAlgorithmParallel::unflatten_matrix(const std::vector<double>& flat, size_t rows, size_t cols) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix[i][j] = flat[i * cols + j];
    }
  }
  return matrix;
}

}  // namespace nasedkin_e_strassen_algorithm_mpi