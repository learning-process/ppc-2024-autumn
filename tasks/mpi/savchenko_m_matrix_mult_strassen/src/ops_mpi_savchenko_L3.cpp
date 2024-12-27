#include "mpi/savchenko_m_matrix_mult_strassen/include/ops_mpi_savchenko_L3.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

// Task Sequential

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  bool valid_icount = (taskData->inputs_count.size() == 1);
  bool valid_ocount = (taskData->outputs_count.size() == 1);
  bool valid_io = (valid_icount && valid_ocount);
  if (!valid_io) return false;

  bool positive_size = (taskData->inputs_count[0] > 0);

  bool valid = (positive_size);
  return valid;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  dims = taskData->inputs_count[0];

  matrix_A = std::vector<double>(dims * dims, 0.0);
  matrix_B = std::vector<double>(dims * dims, 0.0);
  matrix_C = std::vector<double>(dims * dims, 0.0);

  auto* tmp_A = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_A, tmp_A + dims * dims, matrix_A.begin());

  auto* tmp_B = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_B, tmp_B + dims * dims, matrix_B.begin());

  return true;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  matrix_C = strassen(matrix_A, matrix_B, dims);
  return true;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* ptr_C = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(matrix_C.begin(), matrix_C.begin() + dims * dims, ptr_C);
  return true;
}

// Sequential methods

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::is_power_of_two(size_t _size) {
  return (_size != 0) && ((_size & (_size - 1)) == 0);
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::add_matrices(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  size_t _size_vector = _size * _size;
  std::vector<double> C(_size_vector, 0.0);
  for (size_t i = 0; i < _size_vector; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::sub_matrices(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  size_t _size_vector = _size * _size;
  std::vector<double> C(_size_vector, 0.0);
  for (size_t i = 0; i < _size_vector; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::multiply_standard(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  std::vector<double> C(_size * _size, 0.0);
  for (size_t i = 0; i < _size; ++i) {
    for (size_t j = 0; j < _size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < _size; ++k) {
        sum += A[i * _size + k] * B[k * _size + j];
      }
      C[i * _size + j] = sum;
    }
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskSequential::strassen(const std::vector<double>& A,
                                                                                          const std::vector<double>& B,
                                                                                          size_t _size) {
  if (_size <= 4) {  // Border of recursion
    return multiply_standard(A, B, _size);
  }

  size_t size = _size;
  if (!is_power_of_two(_size)) {
    size = 1;
    while (size < _size) size *= 2;
  }
  size_t size_vector = size * size;

  std::vector<double> local_A(size_vector, 0.0);
  std::vector<double> local_B(size_vector, 0.0);

  for (size_t i = 0; i < _size; i++) {
    for (size_t j = 0; j < _size; j++) {
      local_A[i * size + j] = A[i * _size + j];
      local_B[i * size + j] = B[i * _size + j];
    }
  }

  size_t size_half = size / 2;
  size_t size_half_vector = size_half * size_half;

  std::vector<double> A11(size_half_vector, 0.0);
  std::vector<double> A12(size_half_vector, 0.0);
  std::vector<double> A21(size_half_vector, 0.0);
  std::vector<double> A22(size_half_vector, 0.0);

  std::vector<double> B11(size_half_vector, 0.0);
  std::vector<double> B12(size_half_vector, 0.0);
  std::vector<double> B21(size_half_vector, 0.0);
  std::vector<double> B22(size_half_vector, 0.0);

  for (size_t i = 0; i < size_half; ++i) {
    for (size_t j = 0; j < size_half; ++j) {
      A11[i * size_half + j] = local_A[i * size + j];
      A12[i * size_half + j] = local_A[i * size + j + size_half];
      A21[i * size_half + j] = local_A[(i + size_half) * size + j];
      A22[i * size_half + j] = local_A[(i + size_half) * size + j + size_half];

      B11[i * size_half + j] = local_B[i * size + j];
      B12[i * size_half + j] = local_B[i * size + j + size_half];
      B21[i * size_half + j] = local_B[(i + size_half) * size + j];
      B22[i * size_half + j] = local_B[(i + size_half) * size + j + size_half];
    }
  }

  std::vector<std::vector<double>> M(7, std::vector<double>(size_half_vector, 0.0));

  M[0] = strassen(add_matrices(A11, A22, size_half), add_matrices(B11, B22, size_half), size_half);
  M[1] = strassen(add_matrices(A21, A22, size_half), B11, size_half);
  M[2] = strassen(A11, sub_matrices(B12, B22, size_half), size_half);
  M[3] = strassen(A22, sub_matrices(B21, B11, size_half), size_half);
  M[4] = strassen(add_matrices(A11, A12, size_half), B22, size_half);
  M[5] = strassen(sub_matrices(A21, A11, size_half), add_matrices(B11, B12, size_half), size_half);
  M[6] = strassen(sub_matrices(A12, A22, size_half), add_matrices(B21, B22, size_half), size_half);

  std::vector<double> local_C(size_vector, 0.0);

  for (size_t i = 0; i < size_half; ++i) {
    for (size_t j = 0; j < size_half; ++j) {
      local_C[i * size + j] =
          M[0][i * size_half + j] + M[3][i * size_half + j] - M[4][i * size_half + j] + M[6][i * size_half + j];

      local_C[i * size + j + size_half] = M[2][i * size_half + j] + M[4][i * size_half + j];

      local_C[(i + size_half) * size + j] = M[1][i * size_half + j] + M[3][i * size_half + j];

      local_C[(i + size_half) * size + j + size_half] =
          M[0][i * size_half + j] + M[2][i * size_half + j] - M[1][i * size_half + j] + M[5][i * size_half + j];
    }
  }

  std::vector<double> C(_size * _size, 0.0);
  for (size_t i = 0; i < _size; i++) {
    for (size_t j = 0; j < _size; j++) {
      C[i * _size + j] = local_C[i * size + j];
    }
  }

  return C;
}

// Task Parallel

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    bool valid_icount = (taskData->inputs_count.size() == 1);
    bool valid_ocount = (taskData->outputs_count.size() == 1);
    bool valid_io = (valid_icount && valid_ocount);
    if (!valid_io) return false;

    bool positive_size = (taskData->inputs_count[0] > 0);

    bool valid = (positive_size);
    return valid;
  }
  return true;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    dims = taskData->inputs_count[0];

    matrix_A = std::vector<double>(dims * dims, 0.0);
    matrix_B = std::vector<double>(dims * dims, 0.0);
    matrix_C = std::vector<double>(dims * dims, 0.0);

    auto* tmp_A = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp_A, tmp_A + dims * dims, matrix_A.begin());

    auto* tmp_B = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp_B, tmp_B + dims * dims, matrix_B.begin());
  }

  return true;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  matrix_C = strassen_parallel(matrix_A, matrix_B, dims);
  return true;
}

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* ptr_C = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(matrix_C.begin(), matrix_C.begin() + dims * dims, ptr_C);
  }
  return true;
}

// Parallel methods

bool savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::is_power_of_two(size_t _size) {
  return (_size != 0) && ((_size & (_size - 1)) == 0);
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::add_matrices(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  size_t _size_vector = _size * _size;
  std::vector<double> C(_size_vector, 0.0);
  for (size_t i = 0; i < _size_vector; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::sub_matrices(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  size_t _size_vector = _size * _size;
  std::vector<double> C(_size_vector, 0.0);
  for (size_t i = 0; i < _size_vector; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::multiply_standard(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  std::vector<double> C(_size * _size, 0.0);
  for (size_t i = 0; i < _size; ++i) {
    for (size_t j = 0; j < _size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < _size; ++k) {
        sum += A[i * _size + k] * B[k * _size + j];
      }
      C[i * _size + j] = sum;
    }
  }
  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::strassen(const std::vector<double>& A,
                                                                                        const std::vector<double>& B,
                                                                                        size_t _size) {
  if (_size <= 4) {  // Border of recursion
    return multiply_standard(A, B, _size);
  }

  size_t size = _size;
  if (!is_power_of_two(_size)) {
    size = 1;
    while (size < _size) size *= 2;
  }
  size_t size_vector = size * size;

  std::vector<double> local_A(size_vector, 0.0);
  std::vector<double> local_B(size_vector, 0.0);

  for (size_t i = 0; i < _size; i++) {
    for (size_t j = 0; j < _size; j++) {
      local_A[i * size + j] = A[i * _size + j];
      local_B[i * size + j] = B[i * _size + j];
    }
  }

  size_t size_half = size / 2;
  size_t size_half_vector = size_half * size_half;

  std::vector<double> A11(size_half_vector, 0.0);
  std::vector<double> A12(size_half_vector, 0.0);
  std::vector<double> A21(size_half_vector, 0.0);
  std::vector<double> A22(size_half_vector, 0.0);

  std::vector<double> B11(size_half_vector, 0.0);
  std::vector<double> B12(size_half_vector, 0.0);
  std::vector<double> B21(size_half_vector, 0.0);
  std::vector<double> B22(size_half_vector, 0.0);

  for (size_t i = 0; i < size_half; ++i) {
    for (size_t j = 0; j < size_half; ++j) {
      A11[i * size_half + j] = local_A[i * size + j];
      A12[i * size_half + j] = local_A[i * size + j + size_half];
      A21[i * size_half + j] = local_A[(i + size_half) * size + j];
      A22[i * size_half + j] = local_A[(i + size_half) * size + j + size_half];

      B11[i * size_half + j] = local_B[i * size + j];
      B12[i * size_half + j] = local_B[i * size + j + size_half];
      B21[i * size_half + j] = local_B[(i + size_half) * size + j];
      B22[i * size_half + j] = local_B[(i + size_half) * size + j + size_half];
    }
  }

  std::vector<std::vector<double>> M(7, std::vector<double>(size_half_vector, 0.0));

  M[0] = strassen(add_matrices(A11, A22, size_half), add_matrices(B11, B22, size_half), size_half);
  M[1] = strassen(add_matrices(A21, A22, size_half), B11, size_half);
  M[2] = strassen(A11, sub_matrices(B12, B22, size_half), size_half);
  M[3] = strassen(A22, sub_matrices(B21, B11, size_half), size_half);
  M[4] = strassen(add_matrices(A11, A12, size_half), B22, size_half);
  M[5] = strassen(sub_matrices(A21, A11, size_half), add_matrices(B11, B12, size_half), size_half);
  M[6] = strassen(sub_matrices(A12, A22, size_half), add_matrices(B21, B22, size_half), size_half);

  std::vector<double> local_C(size_vector, 0.0);

  for (size_t i = 0; i < size_half; ++i) {
    for (size_t j = 0; j < size_half; ++j) {
      local_C[i * size + j] =
          M[0][i * size_half + j] + M[3][i * size_half + j] - M[4][i * size_half + j] + M[6][i * size_half + j];

      local_C[i * size + j + size_half] = M[2][i * size_half + j] + M[4][i * size_half + j];

      local_C[(i + size_half) * size + j] = M[1][i * size_half + j] + M[3][i * size_half + j];

      local_C[(i + size_half) * size + j + size_half] =
          M[0][i * size_half + j] + M[2][i * size_half + j] - M[1][i * size_half + j] + M[5][i * size_half + j];
    }
  }

  std::vector<double> C(_size * _size, 0.0);
  for (size_t i = 0; i < _size; i++) {
    for (size_t j = 0; j < _size; j++) {
      C[i * _size + j] = local_C[i * size + j];
    }
  }

  return C;
}

std::vector<double> savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel::strassen_parallel(
    const std::vector<double>& A, const std::vector<double>& B, size_t _size) {
  if (world.rank() > 6) {
    world.split(1);
    return {};
  }
  boost::mpi::communicator comm = world.split(0);
  int world_rank = comm.rank();
  int world_size = comm.size();
  boost::mpi::broadcast(comm, _size, 0);

  size_t size = _size;
  if (!is_power_of_two(_size)) {
    size = 1;
    while (size < _size) size *= 2;
  }
  size_t size_vector = size * size;

  std::vector<double> local_A(size_vector, 0.0);
  std::vector<double> local_B(size_vector, 0.0);

  if (world_rank == 0) {
    for (size_t i = 0; i < _size; ++i) {
      for (size_t j = 0; j < _size; ++j) {
        local_A[i * size + j] = A[i * _size + j];
        local_B[i * size + j] = B[i * _size + j];
      }
    }
  }

  boost::mpi::broadcast(comm, local_A, 0);
  boost::mpi::broadcast(comm, local_B, 0);
  boost::mpi::broadcast(comm, size, 0);

  size_t size_half = size / 2;
  size_t size_half_vector = size_half * size_half;

  std::vector<double> A11(size_half_vector, 0.0);
  std::vector<double> A12(size_half_vector, 0.0);
  std::vector<double> A21(size_half_vector, 0.0);
  std::vector<double> A22(size_half_vector, 0.0);

  std::vector<double> B11(size_half_vector, 0.0);
  std::vector<double> B12(size_half_vector, 0.0);
  std::vector<double> B21(size_half_vector, 0.0);
  std::vector<double> B22(size_half_vector, 0.0);

  for (size_t i = 0; i < size_half; ++i) {
    for (size_t j = 0; j < size_half; ++j) {
      size_t ind = i * size + j;
      A11[i * size_half + j] = local_A[ind];
      A12[i * size_half + j] = local_A[ind + size_half];
      A21[i * size_half + j] = local_A[ind + size_half * size];
      A22[i * size_half + j] = local_A[ind + size_half * (size + 1)];

      B11[i * size_half + j] = local_B[ind];
      B12[i * size_half + j] = local_B[ind + size_half];
      B21[i * size_half + j] = local_B[ind + size_half * size];
      B22[i * size_half + j] = local_B[ind + size_half * (size + 1)];
    }
  }

  std::vector<std::vector<double>> M(7, std::vector<double>(size_half_vector, 0.0));

  for (int task = world_rank; task < 7; task += world_size) {
    if (task == 0) {
      M[0] = strassen(add_matrices(A11, A22, size_half), add_matrices(B11, B22, size_half), size_half);
    } else if (task == 1) {
      M[1] = strassen(add_matrices(A21, A22, size_half), B11, size_half);
    } else if (task == 2) {
      M[2] = strassen(A11, sub_matrices(B12, B22, size_half), size_half);
    } else if (task == 3) {
      M[3] = strassen(A22, sub_matrices(B21, B11, size_half), size_half);
    } else if (task == 4) {
      M[4] = strassen(add_matrices(A11, A12, size_half), B22, size_half);
    } else if (task == 5) {
      M[5] = strassen(sub_matrices(A21, A11, size_half), add_matrices(B11, B12, size_half), size_half);
    } else if (task == 6) {
      M[6] = strassen(sub_matrices(A12, A22, size_half), add_matrices(B21, B22, size_half), size_half);
    }
  }

  std::vector<double> global_M(7 * size_half_vector, 0.0);
  for (size_t i = 0; i < M.size(); ++i) {
    boost::mpi::reduce(comm, M[i].data(), M[i].size(), global_M.data() + i * size_half_vector, std::plus(), 0);
  }

  if (world_rank == 0) {
    std::vector<double> local_C(size_vector, 0.0);
    for (size_t i = 0; i < size_half; ++i) {
      for (size_t j = 0; j < size_half; ++j) {
        local_C[i * size + j] = global_M[i * size_half + j] + global_M[3 * size_half_vector + i * size_half + j] -
                                global_M[4 * size_half_vector + i * size_half + j] +
                                global_M[6 * size_half_vector + i * size_half + j];
        local_C[i * size + j + size_half] =
            global_M[2 * size_half_vector + i * size_half + j] + global_M[4 * size_half_vector + i * size_half + j];
        local_C[(i + size_half) * size + j] =
            global_M[1 * size_half_vector + i * size_half + j] + global_M[3 * size_half_vector + i * size_half + j];
        local_C[(i + size_half) * size + j + size_half] =
            global_M[i * size_half + j] - global_M[1 * size_half_vector + i * size_half + j] +
            global_M[2 * size_half_vector + i * size_half + j] + global_M[5 * size_half_vector + i * size_half + j];
      }
    }

    std::vector<double> C(_size * _size);
    for (size_t i = 0; i < _size; ++i) {
      for (size_t j = 0; j < _size; ++j) {
        C[i * _size + j] = local_C[i * size + j];
      }
    }
    return C;
  }
  return {};
}
