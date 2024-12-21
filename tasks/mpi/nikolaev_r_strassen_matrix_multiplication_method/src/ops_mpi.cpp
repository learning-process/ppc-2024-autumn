#include "mpi/nikolaev_r_strassen_matrix_multiplication_method/include/ops_mpi.hpp"

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationSequential::pre_processing() {
  internal_order_test();
  auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);
  size_ = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));
  matrixA_.assign(inputsA, inputsA + size_ * size_);
  matrixB_.assign(inputsB, inputsB + size_ * size_);
  result_.resize(size_ * size_);

  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] && is_square_matrix_size(taskData->inputs_count[0]) &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationSequential::run() {
  internal_order_test();
  result_ = strassen_seq(matrixA_, matrixB_, size_);
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationSequential::post_processing() {
  internal_order_test();
  auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result_.begin(), result_.end(), outputs);
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);
    size_ = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));
    matrixA_.assign(inputsA, inputsA + size_ * size_);
    matrixB_.assign(inputsB, inputsB + size_ * size_);
    result_.resize(size_ * size_);
  }
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->inputs_count[1] && is_square_matrix_size(taskData->inputs_count[0]) &&
           taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationParallel::run() {
  internal_order_test();
  result_ = strassen_mpi(matrixA_, matrixB_, size_);
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.end(), outputs);
  }
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::is_square_matrix_size(size_t n) {
  size_t sqrt_n = static_cast<size_t>(std::sqrt(n));
  return sqrt_n * sqrt_n == n;
}

bool nikolaev_r_strassen_matrix_multiplication_method_mpi::is_power_of_two(size_t n) { return (n && !(n & (n - 1))); }

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_mpi::add(const std::vector<double>& A,
                                                                              const std::vector<double>& B, size_t n) {
  std::vector<double> result(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    result[i] = A[i] + B[i];
  }
  return result;
}

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_mpi::subtract(const std::vector<double>& A,
                                                                                   const std::vector<double>& B,
                                                                                   size_t n) {
  std::vector<double> result(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    result[i] = A[i] - B[i];
  }
  return result;
}

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_mpi::strassen_seq(const std::vector<double>& A,
                                                                                       const std::vector<double>& B,
                                                                                       size_t n) {
  if (n == 1) {
    return {A[0] * B[0]};
  }

  size_t newSize = n;
  if (!is_power_of_two(n)) {
    newSize = 1;
    while (newSize < n) newSize *= 2;
  }

  std::vector<double> A_ext(newSize * newSize, 0.0);
  std::vector<double> B_ext(newSize * newSize, 0.0);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      A_ext[i * newSize + j] = A[i * n + j];
      B_ext[i * newSize + j] = B[i * n + j];
    }
  }

  size_t half = newSize / 2;

  std::vector<double> A11(half * half), A12(half * half), A21(half * half), A22(half * half);
  std::vector<double> B11(half * half), B12(half * half), B21(half * half), B22(half * half);

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      A11[i * half + j] = A_ext[i * newSize + j];
      A12[i * half + j] = A_ext[i * newSize + j + half];
      A21[i * half + j] = A_ext[(i + half) * newSize + j];
      A22[i * half + j] = A_ext[(i + half) * newSize + j + half];

      B11[i * half + j] = B_ext[i * newSize + j];
      B12[i * half + j] = B_ext[i * newSize + j + half];
      B21[i * half + j] = B_ext[(i + half) * newSize + j];
      B22[i * half + j] = B_ext[(i + half) * newSize + j + half];
    }
  }

  auto M1 = strassen_seq(add(A11, A22, half), add(B11, B22, half), half);
  auto M2 = strassen_seq(add(A21, A22, half), B11, half);
  auto M3 = strassen_seq(A11, subtract(B12, B22, half), half);
  auto M4 = strassen_seq(A22, subtract(B21, B11, half), half);
  auto M5 = strassen_seq(add(A11, A12, half), B22, half);
  auto M6 = strassen_seq(subtract(A21, A11, half), add(B11, B12, half), half);
  auto M7 = strassen_seq(subtract(A12, A22, half), add(B21, B22, half), half);

  std::vector<double> result_ext(newSize * newSize, 0.0);

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      result_ext[i * newSize + j] = M1[i * half + j] + M4[i * half + j] - M5[i * half + j] + M7[i * half + j];
      result_ext[i * newSize + j + half] = M3[i * half + j] + M5[i * half + j];
      result_ext[(i + half) * newSize + j] = M2[i * half + j] + M4[i * half + j];
      result_ext[(i + half) * newSize + j + half] =
          M1[i * half + j] + M3[i * half + j] - M2[i * half + j] + M6[i * half + j];
    }
  }

  std::vector<double> result(n * n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      result[i * n + j] = result_ext[i * newSize + j];
    }
  }

  return result;
}

std::vector<double>
nikolaev_r_strassen_matrix_multiplication_method_mpi::StrassenMatrixMultiplicationParallel::strassen_mpi(
    const std::vector<double>& A, const std::vector<double>& B, size_t n) {
  if (world.rank() > 6) {
    world.split(1);
    return {};
  }

  boost::mpi::communicator active_comm = world.split(0);

  int rank = active_comm.rank();
  int size = active_comm.size();

  size_t newSize = n;
  if (!is_power_of_two(n)) {
    newSize = 1;
    while (newSize < n) newSize *= 2;
  }

  std::vector<double> A_ext(newSize * newSize, 0.0);
  std::vector<double> B_ext(newSize * newSize, 0.0);

  if (rank == 0) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        A_ext[i * newSize + j] = A[i * n + j];
        B_ext[i * newSize + j] = B[i * n + j];
      }
    }
  }

  boost::mpi::broadcast(active_comm, A_ext, 0);
  boost::mpi::broadcast(active_comm, B_ext, 0);
  boost::mpi::broadcast(active_comm, newSize, 0);

  size_t half = newSize / 2;

  std::vector<double> A11(half * half, 0.0), A12(half * half, 0.0), A21(half * half, 0.0), A22(half * half, 0.0);
  std::vector<double> B11(half * half, 0.0), B12(half * half, 0.0), B21(half * half, 0.0), B22(half * half, 0.0);

  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      size_t index = i * newSize + j;
      A11[i * half + j] = A_ext[index];
      A12[i * half + j] = A_ext[index + half];
      A21[i * half + j] = A_ext[index + half * newSize];
      A22[i * half + j] = A_ext[index + half * (newSize + 1)];

      B11[i * half + j] = B_ext[index];
      B12[i * half + j] = B_ext[index + half];
      B21[i * half + j] = B_ext[index + half * newSize];
      B22[i * half + j] = B_ext[index + half * (newSize + 1)];
    }
  }

  std::vector<double> M1(half * half, 0.0), M2(half * half, 0.0), M3(half * half, 0.0), M4(half * half, 0.0),
      M5(half * half, 0.0), M6(half * half, 0.0), M7(half * half, 0.0);

  for (int task = rank; task < 7; task += size) {
    switch (task) {
      case 0:
        M1 = strassen_seq(add(A11, A22, half), add(B11, B22, half), half);
        break;
      case 1:
        M2 = strassen_seq(add(A21, A22, half), B11, half);
        break;
      case 2:
        M3 = strassen_seq(A11, subtract(B12, B22, half), half);
        break;
      case 3:
        M4 = strassen_seq(A22, subtract(B21, B11, half), half);
        break;
      case 4:
        M5 = strassen_seq(add(A11, A12, half), B22, half);
        break;
      case 5:
        M6 = strassen_seq(subtract(A21, A11, half), add(B11, B12, half), half);
        break;
      case 6:
        M7 = strassen_seq(subtract(A12, A22, half), add(B21, B22, half), half);
        break;
      default:
        std::cerr << "Process " << rank << ": Skipping unexpected task " << task << std::endl;
    }
  }

  std::vector<double> M1_global(half * half, 0.0), M2_global(half * half, 0.0), M3_global(half * half, 0.0),
      M4_global(half * half, 0.0), M5_global(half * half, 0.0), M6_global(half * half, 0.0),
      M7_global(half * half, 0.0);

  boost::mpi::reduce(active_comm, M1.data(), M1.size(), M1_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M2.data(), M2.size(), M2_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M3.data(), M3.size(), M3_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M4.data(), M4.size(), M4_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M5.data(), M5.size(), M5_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M6.data(), M6.size(), M6_global.data(), std::plus<double>(), 0);
  boost::mpi::reduce(active_comm, M7.data(), M7.size(), M7_global.data(), std::plus<double>(), 0);

  if (rank == 0) {
    std::vector<double> result_ext(newSize * newSize, 0.0);

    for (size_t i = 0; i < half; ++i) {
      for (size_t j = 0; j < half; ++j) {
        result_ext[i * newSize + j] =
            M1_global[i * half + j] + M4_global[i * half + j] - M5_global[i * half + j] + M7_global[i * half + j];
        result_ext[i * newSize + j + half] = M3_global[i * half + j] + M5_global[i * half + j];
        result_ext[(i + half) * newSize + j] = M2_global[i * half + j] + M4_global[i * half + j];
        result_ext[(i + half) * newSize + j + half] =
            M1_global[i * half + j] - M2_global[i * half + j] + M3_global[i * half + j] + M6_global[i * half + j];
      }
    }

    std::vector<double> final_result(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        final_result[i * n + j] = result_ext[i * newSize + j];
      }
    }

    return final_result;
  }
  return {};
}