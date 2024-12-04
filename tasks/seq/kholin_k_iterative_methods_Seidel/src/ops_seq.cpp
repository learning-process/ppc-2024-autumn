#include "seq/kholin_k_iterative_methods_Seidel/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool kholin_k_iterative_methods_Seidel_seq::IsDiagPred(std::vector<float> row_coeffs, size_t num_colls,
                                                       size_t start_index, size_t index) {
  float diag_element = std::fabs(row_coeffs[index]);
  float abs_sum = 0;
  float abs_el = 0;
  size_t size = num_colls;
  for (size_t j = start_index; j < start_index + size; j++) {
    if (j == index) {
      continue;
    }
    abs_el = std::fabs(row_coeffs[j]);
    abs_sum += abs_el;
  }
  return diag_element > abs_sum;
}

int kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::rank(std::vector<float> local_matrix, size_t n,
                                                                    size_t m) {
  int rank = 0;

  for (size_t i = 0; i < std::min(n, m); ++i) {
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(local_matrix[k * m + i]) > std::abs(local_matrix[max_row * m + i])) {
        max_row = k;
      }
    }

    if (std::abs(local_matrix[max_row * m + i]) < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    std::swap(local_matrix[i], local_matrix[max_row]);

    for (size_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j * m + i] / local_matrix[i * m + i];
      for (size_t k = i; k < m; ++k) {
        local_matrix[j * m + k] -= factor * local_matrix[i * m + k];
      }
    }
    rank++;
  }
  return rank;
}

float kholin_k_iterative_methods_Seidel_seq::gen_float_value() {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> coeff(-100, 100);

  return coeff(gen);
}

std::vector<float> kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<float> A_(num_rows * num_colls, 0.0f);
  float p1 = -(1000.0f * 1000.0f * 1000.0f);
  float p2 = -p1;
  float mult = 100 * 100;
  std::uniform_real_distribution<float> coeff_diag(p1, p2);
  std::uniform_real_distribution<float> coeff_no_diag(-10000, 10000);

  for (size_t i = 0; i < num_rows; i++) {
    do {
      for (size_t j = 0; j < num_colls; j++) {
        if (i == j) {
          A_[num_colls * i + j] = mult * coeff_diag(gen);
        } else {
          A_[num_colls * i + j] = coeff_no_diag(gen);
        }
      }
    } while (!IsDiagPred(A_, num_colls, num_colls * i, num_colls * i + i));
  }
  return A_;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  n_rows = taskData->inputs_count[0];
  n_colls = taskData->inputs_count[1];

  AllocateBuffers();

  auto* ptr_vector = reinterpret_cast<float*>(taskData->inputs[0]);
  std::memcpy(A.data(), ptr_vector, sizeof(float) * (n_rows * n_colls));

  auto* ptr = reinterpret_cast<float*>(taskData->inputs[1]);
  epsilon = *ptr;

  auto* ptr_vector_X0 = reinterpret_cast<float*>(taskData->inputs[2]);
  std::memcpy(X0.data(), ptr_vector_X0, sizeof(float) * n_rows);

  iteration_perfomance();
  return true;
}

void kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::AllocateBuffers() {
  X_next = std::vector<float>(n_rows, 0.0f);
  X_prev = std::vector<float>(n_rows, 0.0f);
  X = std::vector<float>(n_rows, 1.0f);
  B = std::vector<float>(n_rows);
  X0 = std::vector<float>(n_rows);
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::validation() {
  internal_order_test();
  auto* matrix = reinterpret_cast<float*>(taskData->inputs[0]);
  auto* B_ = reinterpret_cast<float*>(taskData->inputs[3]);

  size_t num_rows = taskData->inputs_count[0];
  size_t num_colls = taskData->inputs_count[1];
  if (!IsQuadro(num_rows, num_colls)) {
    return false;
  }
  A.assign(matrix, matrix + (num_rows * num_colls));
  if (!CheckDiagPred(matrix, num_rows, num_colls)) {
    return false;
  }
  std::vector<float> matrix_extended(num_rows * num_colls + 1);
  B.assign(B_, B_ + num_rows);
  size_t k = 0;
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_colls; j++) {
      matrix_extended[num_colls + 1 * i + j] = A[num_colls * i + j];
      if (j + 1 == num_colls) {
        k = j + 1;
      }
    }
    matrix_extended[num_colls + 1 * i + k] = B[i];
  }
  int rank_A = rank(A, num_rows, num_colls);
  int rank_A_ = rank(matrix_extended, num_rows, num_colls);
  bool IsSingleDecision = rank_A == rank_A_;
  if (!IsSingleDecision) {
    return IsSingleDecision;
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::run() {
  internal_order_test();
  method_Seidel();
  return true;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<float*>(taskData->outputs[0]);
  std::copy(X.begin(), X.end(), ptr);
  return true;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::IsQuadro(size_t num_rows, size_t num_colls) {
  return num_rows == num_colls;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::CheckDiagPred(float matrix[], size_t num_rows,
                                                                              size_t num_colls) {
  size_t rows = num_rows;
  size_t colls = num_colls;
  float abs_diag_element = 0.0f;
  float abs_el = 0.0f;
  float abs_sum = 0.0f;
  for (size_t i = 0; i < rows; i++) {
    abs_diag_element = std::fabs(matrix[colls * i + i]);
    for (size_t j = 0; j < colls; j++) {
      if (j == i) {
        continue;
      }
      abs_el = std::fabs(matrix[colls * i + j]);
      abs_sum += abs_el;
    }
    if (abs_diag_element <= abs_sum) {
      return false;
    }
    abs_sum = 0;
  }
  return true;
}
std::vector<float> kholin_k_iterative_methods_Seidel_seq::gen_vector(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> coeff(-100.0f, 100.0f);

  std::vector<float> row(sz);

  for (size_t i = 0; i < sz; i++) {
    row[i] = coeff(gen);
  }

  return row;
}
void kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::iteration_perfomance() {
  C = std::vector<float>(n_rows * n_colls, 0.0f);
  for (size_t i = 0; i < n_rows; i++) {
    B[i] = B[i] / A[n_colls * i + i];
    for (size_t j = 0; j < n_colls; j++) {
      if (i == j) {
        C[n_colls * i + i] = 0.0f;
        continue;
      }
      C[n_colls * i + j] = -A[n_colls * i + j] / A[n_colls * i + i];
    }
  }

  std::copy(B.begin(), B.end(), X0.begin());
}

float kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::d() {
  // AX + B < epsilon
  float d = 0;
  float maxd = 0;
  for (size_t i = 0; i < n_rows; i++) {
    d = std::fabs(X_next[i] - X_prev[i]);
    if (d > maxd) {
      maxd = d;
    }
  }
  return maxd;
}

void kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::method_Seidel() {
  float delta = 1.0f;
  for (size_t k_iteration = 0; delta > epsilon; k_iteration++) {
    for (size_t i = 0; i < n_rows; i++) {
      for (size_t j = 0; j < n_colls; j++) {
        if (j < i) {
          X_next[i] += C[n_colls * i + j] * X_next[j];
        } else if (j > i) {
          if (k_iteration == 0) {
            X_next[i] += C[n_colls * i + j] * X0[j];
          } else {
            X_next[i] += C[n_colls * i + j] * X_prev[j];
          }
        }
      }
      X_next[i] += B[i];
    }
    delta = d();
    std::copy(X_next.begin(), X_next.end(), X_prev.begin());
    std::fill(X_next.begin(), X_next.end(), 0.0f);
  }
  std::copy(X_prev.begin(), X_prev.end(), X.begin());
}
