#include "seq/khokhlov_a_iterative_seidel_method/include/ops_seq_khokhlov.hpp"

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::pre_processing() {
  internal_order_test();
  // init matrix
  A = std::vector<double>(taskData->inputs_count[0] * taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp, tmp + taskData->inputs_count[0] * taskData->inputs_count[0], A.begin());

  // init vector
  b = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp1 = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp1, tmp1 + taskData->inputs_count[0], b.begin());

  // init sizes
  n = taskData->inputs_count[0];

  // init maxIterations
  maxIterations = taskData->inputs_count[1];

  EPSILON = taskData->inputs_count[2];

  // Init value for output
  result = std::vector<double>(taskData->inputs_count[0], 0);
  return true;
}
bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::validation() {
  internal_order_test();
  int N = taskData->inputs_count[0];
  int iter = taskData->inputs_count[1];
  if (N < 1) return false;
  if (iter < 1) return false;
  std::vector<double> A_ = std::vector<double>(N * N);
  auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp, tmp + N * N, A_.begin());
  std::vector<double> b_ = std::vector<double>(N);
  tmp = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp, tmp + N, b_.begin());
  for (int i = 0; i < N; i++) {
    if (A_[i * N + i] == 0) return false;
  }
  std::vector<double> Ab = std::vector<double>(N * (N + 1));
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N + 1; j++) {
      Ab[i * N + j] = (j % N == 0) ? b_[i] : A_[i * N + j];
    }
  int rankA = rank(A_, N, N);
  int rankAb = rank(Ab, N, N + 1);
  return (taskData->inputs_count[3] >= 0 && rankA == rankAb);
}

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::run() {
  internal_order_test();
  std::vector<double> x(n, 1.0);
  std::vector<double> prevX(n, 1.0);

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          sum += (A[i * n + j] * x[j]);
        }
      }
      if (A[i * n + i] != 0) {
        x[i] = (b[i] - sum) / A[i * n + i];
      } else {
        x[i] = 0.0;
      }
    }
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
      norm += std::pow(x[i] - prevX[i], 2);
    }
    norm = std::sqrt(norm);

    if (norm < EPSILON) {
      break;
    }
    prevX = x;
  }
  result = x;
  return true;
}

bool khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::post_processing() {
  internal_order_test();
  for (int i = 0; i < n; i++) reinterpret_cast<double*>(taskData->outputs[0])[i] = result[i];
  return true;
}

int khokhlov_a_iterative_seidel_method_seq::seidel_method_seq::rank(std::vector<double> A_, int rows, int cols) {
  int rank = 0;
  for (int i = 0; i < std::min(rows, cols); ++i) {
    int maxRow = i;
    for (int k = i + 1; k < rows; ++k) {
      if (std::abs(A_[k * rows + i]) > std::abs(A_[maxRow * rows + i])) {
        maxRow = k;
      }
    }
    if (std::abs(A_[maxRow * rows + i]) < 1e-9) {
      continue;
    }
    std::swap(A_[i], A_[maxRow]);
    for (int j = i + 1; j < rows; ++j) {
      double factor = A_[j * rows + i] / A_[i * rows + i];
      for (int k = i; k < cols; ++k) {
        A_[j * rows + k] -= factor * A_[i * rows + k];
      }
    }
    rank++;
  }
  return rank;
}