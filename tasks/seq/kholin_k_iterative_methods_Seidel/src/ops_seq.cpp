#include "seq/kholin_k_iterative_methods_Seidel/include/ops_seq.hpp"

#include <memory.h>

#include <thread>
//

using namespace std::chrono_literals;

namespace kholin_k_iterative_methods_Seidel_seq {
float* A_;
}

void kholin_k_iterative_methods_Seidel_seq::freeA_() {
  delete[] A_;
  A_ = nullptr;
}

void kholin_k_iterative_methods_Seidel_seq::copyA_(float val[], size_t num_rows, size_t num_colls) {
  std::memcpy(val, A_, sizeof(float) * num_rows * num_colls);
}
void kholin_k_iterative_methods_Seidel_seq::setA_(float val[], size_t num_rows, size_t num_colls) {
  A_ = new float[num_rows * num_colls];
  std::memcpy(A_, val, sizeof(float) * num_rows * num_colls);
}

float*& kholin_k_iterative_methods_Seidel_seq::getA_() { return A_; }

bool kholin_k_iterative_methods_Seidel_seq::IsDiagPred(float row_coeffs[], size_t num_colls, size_t start_index,
                                                       size_t index) {
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

float kholin_k_iterative_methods_Seidel_seq::gen_float_value() {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> coeff(-100, 100);

  return coeff(gen);
}

bool kholin_k_iterative_methods_Seidel_seq::gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls) {
  std::random_device dev;
  std::mt19937 gen(dev());
  A_ = new float[num_rows * num_colls];
  std::fill(A_, A_ + num_rows * num_colls, 0.0f);
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
  return true;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  n_rows = taskData->inputs_count[0];
  n_colls = taskData->inputs_count[1];
  A = new float[n_rows * n_colls];
  auto* ptr_vector = reinterpret_cast<float*>(taskData->inputs[0]);
  std::memcpy(A, ptr_vector, sizeof(float) * (n_rows * n_colls));
  X_next = new float[n_rows];
  std::fill(X_next, X_next + n_rows, 0.0f);
  X_prev = new float[n_rows];
  std::fill(X_prev, X_prev + n_rows, 0.0f);
  B = gen_vector(n_rows);
  auto* ptr = reinterpret_cast<float*>(taskData->inputs[1]);
  epsilon = *ptr;
  X = new float[n_rows];
  std::fill(X, X + n_rows, 1.0f);
  X0 = new float[n_rows];
  iteration_perfomance();
  return true;
}

void kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::SetDefault() {
  A = nullptr;
  X = nullptr;
  X_next = nullptr;
  X_prev = nullptr;
  X0 = nullptr;
  B = nullptr;
  C = nullptr;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::validation() {
  internal_order_test();
  SetDefault();
  return CheckDiagPred(getA_(), taskData->inputs_count[0], taskData->inputs_count[1]) &&
         IsQuadro(taskData->inputs_count[0], taskData->inputs_count[1]);
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::run() {
  internal_order_test();
  method_Seidel();
  return true;
}

bool kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* ptr = reinterpret_cast<float*>(taskData->outputs[0]);
  std::memcpy(ptr, X, sizeof(float) * n_rows);
  return true;
}

kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::~TestTaskSequential() {
  delete[] A;
  delete[] X;
  delete[] X_next;
  delete[] X_prev;
  delete[] X0;
  delete[] B;
  delete[] C;
  delete[] A_;
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

float* kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::gen_vector(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  auto* row = new float[sz];
  std::uniform_real_distribution<float> coeff(-100, 100);
  for (size_t i = 0; i < sz; i++) {
    row[i] = coeff(gen);
  }
  return row;
}

void kholin_k_iterative_methods_Seidel_seq::TestTaskSequential::iteration_perfomance() {
  // X = CX` + B
  C = new float[n_rows * n_colls];
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_colls; j++) {
      if (i == j) {
        B[i] = B[i] / A[n_colls * i + i];
        C[n_colls * i + i] = 0;
        continue;
      }
      C[n_colls * i + j] = -A[n_colls * i + j] / A[n_colls * i + i];
    }
  }
  std::memcpy(X0, B, n_colls * sizeof(float));
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
    std::memcpy(X_prev, X_next, sizeof(float) * n_rows);
    std::fill(X_next, X_next + n_rows, 0.0f);
  }
  std::memcpy(X, X_prev, sizeof(float) * n_rows);
}
