#include "seq/nikolaev_r_strassen_matrix_multiplication_method/include/ops_seq.hpp"

bool nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential::pre_processing() {
  internal_order_test();

  auto* inputsA = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* inputsB = reinterpret_cast<double*>(taskData->inputs[1]);

  size_ = static_cast<size_t>(std::sqrt(taskData->inputs_count[0]));
  matrixA_.assign(inputsA, inputsA + size_ * size_);
  matrixB_.assign(inputsB, inputsB + size_ * size_);
  result_.resize(size_ * size_);

  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential::validation() {
  internal_order_test();
  return !taskData->inputs.empty() && taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == static_cast<size_t>(std::sqrt(taskData->inputs_count[0])) *
                                          static_cast<size_t>(std::sqrt(taskData->inputs_count[0])) &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential::run() {
  internal_order_test();
  result_ = nikolaev_r_strassen_matrix_multiplication_method_seq::strassen(matrixA_, matrixB_, size_);
  return true;
}

bool nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential::post_processing() {
  internal_order_test();

  auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);

  std::copy(result_.begin(), result_.end(), outputs);

  return true;
}

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_seq::add(const std::vector<double>& A,
                                                                              const std::vector<double>& B, size_t n) {
  std::vector<double> result(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    result[i] = A[i] + B[i];
  }
  return result;
}

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_seq::subtract(const std::vector<double>& A,
                                                                                   const std::vector<double>& B,
                                                                                   size_t n) {
  std::vector<double> result(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    result[i] = A[i] - B[i];
  }
  return result;
}

std::vector<double> nikolaev_r_strassen_matrix_multiplication_method_seq::strassen(const std::vector<double>& A,
                                                                                   const std::vector<double>& B,
                                                                                   size_t n) {
  if (n == 1) {
    return {A[0] * B[0]};
  }

  size_t newSize = n;
  if ((n == 0) || ((n & (n - 1)) != 0)) {
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
  size_t half_squared = half * half;

  std::vector<double> A11(half_squared);
  std::vector<double> A12(half_squared);
  std::vector<double> A21(half_squared);
  std::vector<double> A22(half_squared);
  std::vector<double> B11(half_squared);
  std::vector<double> B12(half_squared);
  std::vector<double> B21(half_squared);
  std::vector<double> B22(half_squared);

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

  auto M1 = strassen(add(A11, A22, half), add(B11, B22, half), half);
  auto M2 = strassen(add(A21, A22, half), B11, half);
  auto M3 = strassen(A11, subtract(B12, B22, half), half);
  auto M4 = strassen(A22, subtract(B21, B11, half), half);
  auto M5 = strassen(add(A11, A12, half), B22, half);
  auto M6 = strassen(subtract(A21, A11, half), add(B11, B12, half), half);
  auto M7 = strassen(subtract(A12, A22, half), add(B21, B22, half), half);

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
